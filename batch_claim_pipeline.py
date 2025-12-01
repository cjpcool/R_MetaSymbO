"""
Batch Claim → Generation → UMA Optimization → DFT → Analysis Pipeline
====================================================================
---------------
Reads a JSONL file (one claim per line). For each claim:
  1. Run gen_test.py with the claim as prompt (LLM or offline depending on arguments)
  2. Run wrap_md_uma.py to perform UMA (and optional DFT) optimization on the generated structure
  3. Run analyze_optimization.py to extract optimization metrics
  4. Record timings for generation / UMA / DFT; write per-claim + global summary JSON/CSV

Output Structure (per claim under --output-root):
  claim_<index>_
    |-- raw_claim.txt
    |-- gen/
    |     generation_time.txt (from gen_test) + generated npz
    |-- mdopt/ (output of wrap_md_uma.py)
    |     timing_log.txt / orca_sp/timing_log.txt
    |     best.traj, best.xyz, summary.json, orca_sp/* if DFT performed
    |-- analysis/
    |     metrics.json, metrics.csv, figures
    |-- timings.json (consolidated times for this claim)

Global summary files under --output-root:
  pipeline_summary.json
  pipeline_summary.csv

JSONL Input Format:
  Each line: {"claim": "text ..."}  (default key configurable via --claim-key)
  Lines that are blank or missing the key are skipped with a warning.

Usage Example:
  python batch_claim_pipeline.py \
      --api-key <API_KEY> \
      --jsonl sprint1-drop4-problems.jsonl \
      --ase-omat /home/grads/jianpengc/datasets/omat24/rattled-relax \
      --ckpt-gen ./checkpoints/omat24_rattle2 \
      --uma-ckpt ./checkpoints/uma-s-1p1.pt \
      --orca-command ~/orca_6_1_0/orca --dft --nprocs 16 --maxcore 6000 \
      --output-root ./batch_runs --preset quick --limit 5

Design Notes:
  • Uses subprocess for isolation; each step timed via time.time() wrapper.
  • Attempts to parse timing files (generation_time.txt, timing_log.txt) for more precise DFT timings.
  • Robust error handling: logs errors, continues to next claim unless --fail-fast.

"""
import os
import sys
import json
import csv
import time
import argparse
import subprocess
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple



# ---------------- Utility ----------------

def read_jsonl(path: str, claim_key: str = "claim") -> List[str]:
    claims = []
    with open(path, "r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception as e:
                print(f"[WARN] Line {ln} not valid JSON: {e}")
                continue
            if claim_key not in obj:
                print(f"[WARN] Line {ln} missing key '{claim_key}', skipped")
                continue
            val = str(obj[claim_key]).strip()
            if val:
                claims.append(val)
            else:
                print(f"[WARN] Line {ln} empty claim text")
    return claims

def run_cmd(cmd: List[str], cwd: Optional[str] = None, log_path: Optional[Path] = None) -> Tuple[int, float, str]:
    start = time.time()
    try:
        proc = subprocess.run(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        elapsed = time.time() - start
        output = proc.stdout
        if log_path:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_path.write_text(output, encoding='utf-8')
        if proc.returncode != 0:
            print(f"[ERROR] Command failed ({proc.returncode}): {' '.join(cmd)}")
        return proc.returncode, elapsed, output
    except Exception as e:
        elapsed = time.time() - start
        msg = f"[EXCEPTION] running {' '.join(cmd)}: {e}"
        if log_path:
            log_path.write_text(msg, encoding='utf-8')
        return 1, elapsed, msg


def parse_generation_time(gen_dir: Path) -> Optional[float]:
    f = gen_dir / 'generation_time.txt'
    if not f.is_file():
        return None
    try:
        lines = f.read_text(encoding='utf-8').strip().splitlines()
        if not lines:
            return None
        # take last line numeric seconds at end
        import re
        for line in reversed(lines):
            m = re.search(r"([0-9]+\.[0-9]+)\s*seconds", line)
            if m:
                return float(m.group(1))
    except Exception:
        return None
    return None

def parse_orca_time(mdopt_dir: Path) -> Optional[float]:
    # attempt to parse ORCA DFT Single-Point time from timing logs
    candidates = [mdopt_dir / 'orca_sp' / 'timing_log.txt', mdopt_dir / 'timing_log.txt', mdopt_dir / 'orca_sp' / 'generation_time.txt']
    import re
    for f in candidates:
        if f.is_file():
            try:
                for line in f.read_text(encoding='utf-8').splitlines():
                    if 'ORCA DFT Single-Point' in line:
                        m = re.search(r'([0-9]+\.[0-9]+)\s*seconds', line)
                        if m:
                            return float(m.group(1))
            except Exception:
                pass
    return None


def write_json(data: Any, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def append_csv(path: Path, header: List[str], row: List[Any]):
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    with open(path, 'a', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        if new:
            w.writerow(header)
        w.writerow(row)

# ---------------- Core Pipeline ----------------

def process_claim(idx: int, claim: str, args) -> Dict[str, Any]:
    claim_dir = Path(args.output_root) / f"claim_{idx:04d}"
    gen_dir = claim_dir / 'gen'
    md_dir = claim_dir / 'mdopt'
    analysis_dir = claim_dir / 'analysis'
    claim_dir.mkdir(parents=True, exist_ok=True)
    (claim_dir / 'raw_claim.txt').write_text(claim, encoding='utf-8')

    # 1) Generation
    gen_cmd = [
        sys.executable, 'gen_test.py',
        '--prompt', claim,
        '--save-dir', str(gen_dir),
        '--ckpt-dir', args.ckpt_gen,
        '--logic-mode', args.logic_mode,
        '--cutoff', str(args.cutoff),
    ]
    if args.api_key:
        gen_cmd += ['--api-key', args.api_key]
    else:
        gen_cmd += ['--api-key', '']
    if args.ase_omat:
        gen_cmd += ['--ase-omat', args.ase_omat]
    if args.no_llm:
        gen_cmd += ['--no-llm']
    if args.prototype:
        gen_cmd += ['--prototype', args.prototype]
    if args.no_generator:
        gen_cmd += ['--no-generator']

    rc_gen, wall_gen, out_gen = run_cmd(gen_cmd, log_path=gen_dir / 'gen_stdout.log')
    gen_time_logged = parse_generation_time(gen_dir)
    if gen_time_logged is None:
        gen_time_logged = wall_gen

    # Determine generated npz (pick latest *.npz)
    gen_npz = None
    if gen_dir.is_dir():
        candidates = sorted(gen_dir.glob('*.npz'), key=lambda p: p.stat().st_mtime, reverse=True)
        if candidates:
            gen_npz = candidates[0]

    # 2) UMA + DFT
    dft_flag = '--run-dft' if args.dft else ''
    uma_cmd = [
        sys.executable, 'wrap_md_uma.py',
        '--gen-path', str(gen_npz) if gen_npz else 'MISSING',
        '--outdir', str(md_dir),
        '--preset', args.preset,
        '--loops', str(args.loops),
        '--fmax', str(args.fmax),
    ]
    if args.uma_ckpt:
        uma_cmd += ['--ckpt', args.uma_ckpt]
    if args.device:
        uma_cmd += ['--device', args.device]
    if args.relax_cell:
        uma_cmd += ['--relax-cell']
    if args.fallback_emt:
        uma_cmd += ['--fallback-emt']
    if args.dft:
        uma_cmd.append('--run-dft')
        if args.orca_command:
            uma_cmd += ['--orca-command', args.orca_command]
        uma_cmd += ['--nprocs', str(args.nprocs), '--maxcore', str(args.maxcore)]
        if args.orcasimpleinput:
            uma_cmd += ['--orcasimpleinput', args.orcasimpleinput]
        if args.orca_extra_blocks:
            uma_cmd += ['--orcablocks-extra', args.orca_extra_blocks]

    rc_uma, wall_uma, out_uma = run_cmd(uma_cmd, log_path=md_dir / 'uma_stdout.log')

    # Parse ORCA time (if any)
    orca_time = parse_orca_time(md_dir) if args.dft else None

    # 3) Analysis
    analysis_cmd = [
        sys.executable, 'structure_optim_modules/analyze_optimization.py',
        '--root', str(md_dir),
        '--pattern', 'loop_*.traj',
        '--save-dir', str(analysis_dir),
        '--export-json', str(analysis_dir / 'metrics.json'),
        '--export-csv', str(analysis_dir / 'metrics.csv'),
        '--no-show', '--compare'
    ]
    rc_analysis, wall_analysis, out_analysis = run_cmd(analysis_cmd, log_path=analysis_dir / 'analysis_stdout.log')

    uma_energy = None
    dft_energy = None
    dft_gap = None
    uma_summary = md_dir / 'summary.json'
    if uma_summary.is_file():
        try:
            data = json.loads(uma_summary.read_text(encoding='utf-8'))
            uma_energy = data.get('uma', {}).get('energy_eV')
            dft_energy = data.get('dft', {}).get('energy_eV') if data.get('dft') else None
            dft_gap = data.get('dft', {}).get('gap_eV') if data.get('dft') else None
        except Exception:
            pass

    result = {
        'index': idx,
        'claim': claim,
        'paths': {
            'claim_dir': str(claim_dir),
            'gen_dir': str(gen_dir),
            'md_dir': str(md_dir),
            'analysis_dir': str(analysis_dir),
            'generated_npz': str(gen_npz) if gen_npz else None,
            'uma_summary': str(uma_summary) if uma_summary.is_file() else None,
        },
        'return_codes': {
            'gen': rc_gen,
            'uma': rc_uma,
            'analysis': rc_analysis,
        },
        'timing_seconds': {
            'gen_wall': wall_gen,
            'gen_logged': gen_time_logged,
            'uma_wall': wall_uma,
            'dft_logged': orca_time,
            'analysis_wall': wall_analysis,
        },
        'energies': {
            'uma_energy_eV': uma_energy,
            'dft_energy_eV': dft_energy,
            'dft_gap_eV': dft_gap,
        }
    }

    write_json(result, claim_dir / 'timings.json')
    return result

# ---------------- CLI ----------------
"""
python batch_claim_pipeline.py --jsonl ./sprint1-drop4-problems.jsonl --start-index 10 --no-generator  --output-root ./batch_runs \
    --api-key '' \
    --ckpt-gen ./checkpoints/omat24_rattle2 \
    --uma-ckpt ./checkpoints/uma-s-1p1.pt \
    --orca-command ~/orca_6_1_0/orca --dft --nprocs 16 --maxcore 4000 \
    --preset quick \
    --fallback-emt
"""
def main():
    ap = argparse.ArgumentParser(description='Batch pipeline for claim → generation → optimization → DFT → analysis.')
    ap.add_argument('--jsonl', required=True, help='Input JSONL with claims')
    ap.add_argument('--claim-key', default='claim', help='Key for claim text in JSON lines (default: claim)')
    ap.add_argument('--output-root', default='./batch_runs', help='Root output directory')
    ap.add_argument('--limit', type=int, default=None, help='Limit number of claims processed')
    ap.add_argument('--start-index', type=int, default=0, help='Start index offset for claim numbering')
    ap.add_argument('--fail-fast', action='store_true', help='Stop on first error')

    # Generation args
    ap.add_argument('--no-generator', action='store_true', help='Disable the generation step of Agent 2')
    ap.add_argument('--api-key', default='')
    ap.add_argument('--ase-omat', default=None)
    ap.add_argument('--ckpt-gen', required=True, help='Checkpoint dir for generation model')
    ap.add_argument('--logic-mode', default='union')
    ap.add_argument('--cutoff', type=float, default=6.0)
    ap.add_argument('--no-llm', action='store_true')
    ap.add_argument('--prototype', default=None)

    # UMA / DFT args
    ap.add_argument('--uma-ckpt', default=None)
    ap.add_argument('--preset', default='quick', choices=['quick','standard','thorough'])
    ap.add_argument('--loops', type=int, default=5)
    ap.add_argument('--fmax', type=float, default=0.03)
    ap.add_argument('--relax-cell', action='store_true')
    ap.add_argument('--fallback-emt', action='store_true')
    ap.add_argument('--device', default=None)
    ap.add_argument('--dft', action='store_true', help='Enable DFT step in UMA pipeline')
    ap.add_argument('--orca-command', default=None)
    ap.add_argument('--nprocs', type=int, default=max(1, os.cpu_count() or 1))
    ap.add_argument('--maxcore', type=int, default=4000)
    ap.add_argument('--orcasimpleinput', default=None)
    ap.add_argument('--orca-extra-blocks', dest='orca_extra_blocks', default=None)

    args = ap.parse_args()

    claims = read_jsonl(args.jsonl, claim_key=args.claim_key)
    if args.limit:
        claims = claims[:args.limit]
    os.makedirs(args.output_root, exist_ok=True)

    summary: List[Dict[str, Any]] = []

    for i, claim in enumerate(claims, args.start_index):
        print(f"\n===== Processing claim {i} / {len(claims)+args.start_index-1} =====")
        try:
            res = process_claim(i, claim, args)
            summary.append(res)
            if (res['return_codes']['gen'] != 0 or res['return_codes']['uma'] != 0) and args.fail_fast:
                print('[FATAL] Step failed and --fail-fast enabled. Stopping.')
                break
        except KeyboardInterrupt:
            print('Interrupted by user.')
            break
        except Exception as e:
            print(f"[ERROR] Unexpected exception for claim index {i}: {e}")
            if args.fail_fast:
                break
            continue

    # Global summary
    global_json = Path(args.output_root) / 'pipeline_summary.json'
    write_json(summary, global_json)

    # CSV summary
    csv_path = Path(args.output_root) / 'pipeline_summary.csv'
    header = ['index','gen_logged','gen_wall','uma_wall','dft_logged','analysis_wall','uma_energy_eV','dft_energy_eV','dft_gap_eV','generated_npz']
    for item in summary:
        append_csv(csv_path, header, [
            item['index'],
            item['timing_seconds'].get('gen_logged'),
            item['timing_seconds'].get('gen_wall'),
            item['timing_seconds'].get('uma_wall'),
            item['timing_seconds'].get('dft_logged'),
            item['timing_seconds'].get('analysis_wall'),
            item['energies'].get('uma_energy_eV'),
            item['energies'].get('dft_energy_eV'),
            item['energies'].get('dft_gap_eV'),
            item['paths'].get('generated_npz'),
        ])

    print(f"\n[DONE] Processed {len(summary)} claims. Global summary written to: {global_json}")

if __name__ == '__main__':
    main()
