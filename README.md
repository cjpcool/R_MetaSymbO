# Overall Components

![alt text](image-1.png)


This repository ties together three phases:

1) **Generation** (Step 0 -> Step 1): Input claims, R_MetaSymbO will generate crystall structures.
    * Entity Extraction
    * Revised MetaSymbO

2) **Relaxation / Optimization** (Step 2): Run an **MD anneal → quench → 0 K relax** loop with a **Fairchem/UMA** calculator.

3) **DFT Characterization** (Step 3): Single-point **ORCA** calculation (SP + EnGrad) on a **non-periodic copy** of the relaxed structure to extract energy, dipole, HOMO-LUMO gap (when available), plus forces.

---

## What’s in here

- **`gen_test.py`** — Generation or offline test for **Step 0 -> Step 1** generation.  
  - Two scaffold modes: **LLM** (Agent-1) and **offline** prototypes (rocksalt/diamond).  
  - Saves structures to `./_gens/*.npz` with keys: `atom_types (Z)`, `cart_coords (Å)`, `lengths [a,b,c] (Å)`, `angles [α,β,γ] (deg)`. 

- **`wrap_md_uma.py`** (Step 2 and Step 3) — wrapper that **reads a generated `.npz/.npy`**, attaches a **UMA** calculator, and calls `optimize_with_md_loop` with a **safe fallback** if your local function exits early. Includes **device auto-detect**, **preset** anneal profiles, and clear EMT fallback behavior.

   **`wrap_md_uma.py`** — full pipeline **(API + CLI)**:  
  - UMA MD/relax + saving `.traj/.xyz/energy.txt`  
  - Optional **ORCA** SP + EnGrad with upgraded defaults (`M062X 6-31G* SP EnGrad D3BJ def2/J RIJCOSX TightSCF NoAutoStart MiniPrint NoPop`)  
  - Light parser for `.out` (final energy, gap, dipole, Mulliken block if present)  
  - Returns a consolidated `{'uma': {...}, 'dft': {...}}` dict and writes `summary.json`. :contentReference[oaicite:1]{index=1}

---

## Quickstart

### 0) Environment

- **Python** ≥ 3.9  
- **PyTorch** + **PyG** (per your GeomVAE)  
- **ASE** (`pip install ase`)  
- **Fairchem / OCP** (UMA) — one of:
  - `pip install fairchem`
  - Download UMA Checkpoint from huggingface
- **ORCA** (optional, for DFT): install ORCA and ensure the binary is on `PATH` or set `ORCA_COMMAND=/path/to/orca`.

### 1) Generate a crystal (Agent-1/2 only)

**LLM scaffold mode (Agent-1):**
Download R_MetaSymbo checkpoint (omat24_rattle2) from https://drive.google.com/drive/folders/1JQ6-tAcz7B5CCfuJSiCyuYng-0eFO9GY?usp=sharing

```bash
python gen_test.py \
  --cif-dir ./data/cifs \
  --ckpt-dir ./checkpoints/omat24_rattle2 \
  --prompt "Wide-gap semiconductor, wurtzite-like scaffold" \
  --save-dir ./_gens
```

**Offline scaffold (no LLM):**
```
python gen_test.py \
  --no-llm --prototype rocksalt \
  --ckpt-dir ./checkpoints/omat24_rattle2 \
  --save-dir ./_gens
```
Output: ./_gens/gen_union.npz (or rocksalt_union_offline.npz) with Z, cart_coords, lengths, angles.

### 2) UMA MD/relax only
Using **UMA (preferred)**. If you don’t pass a checkpoint, script will check `FAIRCHEM_UMA_CKPT`:
```
export FAIRCHEM_UMA_CKPT=/path/to/uma.pt
export FAIRCHEM_UMA_CONFIG=/path/to/config.yaml   # optional
python wrap_md_uma.py \
  --gen-path ./_gens/gen_union.npz \
  --preset standard \
  --outdir ./_mdopt
  ```
**Smoke test** without UMA (EMT fallback):、
```
python wrap_md_uma.py \
  --gen-path ./_gens/rocksalt_union_offline.npz \
  --fallback-emt \
  --preset quick \
  --outdir ./_mdopt_test
```
**Outputs:**

`_mdopt/best.traj`, `_mdopt/best_energy.txt` and logs (when enabled).

### Analyze optimization trajectories
```
python structure_optim_modules/analyze_optimization.py --root ./_mdopt --pattern "loop_*.traj" --save-dir ./_mdopt/analysis \
  --export-json ./_mdopt/analysis/metrics.json --export-csv ./_mdopt/analysis/metrics.csv --no-show --compare
```
**Outputs:**

Analysis results of optimization steps, includding energy change, visulized structure, etc.


### 3) UMA + ORCA-DFT (one-shot)
```
# env-based defaults are supported too: ORCA_COMMAND, FAIRCHEM_UMA_CKPT, FAIRCHEM_UMA_CONFIG
python wrap_md_uma_dft.py \
  --gen-path ./_gens/gen_union.npz \
  --run-dft \
  --nprocs 8 --maxcore 4000 \
  --preset standard \
  --outdir ./_mdopt
```
**Outputs:**
Outputs:

- UMA: _mdopt/best.traj, _mdopt/best.xyz, _mdopt/best_energy.txt

- DFT: _mdopt/orca_sp/ with orca_calc.inp/out, forces.npy, dft_results.json

- Summary: _mdopt/summary.json (merged UMA/DFT fields).


## Programmatic API

You can call the full pipeline from Python and get a single results dict:
```
from wrap_md_uma_dft import optimize_and_characterize

res = optimize_and_characterize(
    gen_path="./_gens/gen_union.npz",
    outdir="./_mdopt",
    uma_ckpt="/path/to/uma.pt",          # or env FAIRCHEM_UMA_CKPT
    uma_config_yml="/path/to/config.yaml",  # optional; or env FAIRCHEM_UMA_CONFIG
    device="cuda",
    preset="standard",                   # quick / standard / thorough
    run_dft=True,
    orca_command="orca",                 # or env ORCA_COMMAND
    orca_maxcore=4000,
    orca_nprocs=8,
    orca_simpleinput=(
        "M062X 6-31G* SP EnGrad D3BJ def2/J RIJCOSX TightSCF NoAutoStart MiniPrint NoPop"
    ),
)
print(res["uma"]["energy_eV"], res.get("dft", {}).get("energy_eV"))
```
This calls UMA MD/relax and optionally ORCA SP→ returns {'uma': {...}, 'dft': {...}}.

## ORCA Settings (DFT Single-Point)
The wrapper uses a robust default derived from your template:
```
M062X 6-31G* SP EnGrad D3BJ def2/J RIJCOSX TightSCF NoAutoStart MiniPrint NoPop
%maxcore {maxcore}
%pal nprocs {nprocs} end
```
## Presets and Defaults

Both wrappers support MD presets:

- quick: 2 loops, milder temperatures; looser fmax for smoke tests.

- standard: sensible defaults (5 loops, 900 K peak, etc.).

- thorough: ≥8 loops, 1400 K peak, slower cool, tighter fmax.

Device auto-detection (cuda if available) and env-driven UMA/ORCA paths:

- FAIRCHEM_UMA_CKPT, FAIRCHEM_UMA_CONFIG, ORCA_COMMAND.