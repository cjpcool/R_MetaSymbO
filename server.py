"""
FastAPI server exposing R_MetaSymbO crystal generation + UMA/DFT optimization.
"""
# pip install fastapi uvicorn pydantic
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# === Optional: import the programmatic pipeline for UMA + ORCA ===
# Make sure your PYTHONPATH can find wrap_md_uma_dft.py
try:
    from wrap_md_uma import optimize_and_characterize  # noqa: F401
    HAS_OPTIMIZER = True
except Exception as e:
    HAS_OPTIMIZER = False
    _IMPORT_ERR = e

app = FastAPI(
    title="R_MetaSymbO Crystal Pipeline API",
    description="Expose generation (LLM scaffold/offline) and UMA/DFT optimization via HTTP.",
    version="0.1.0",
)

# ---------- Models ----------
class GenerateIn(BaseModel):
    claims: str  # the prompt/claims for generation
    ckpt_dir: str
    cif_dir: Optional[str] = None
    save_dir: str = "./_gens"
    use_llm: bool = True  # if False, choose a prototype
    prototype: Optional[str] = None  # "rocksalt" / "diamond" when use_llm=False
    api_key: Optional[str] = None  # if None, use env api_key
    extra_args: Optional[List[str]] = None  # passthrough CLI args, e.g. ["--foo","bar"]

class GenerateOut(BaseModel):
    gen_path: str
    saved_files: List[str]
    log: Optional[str] = None

class OptimizeIn(BaseModel):
    gen_path: str
    outdir: str = "./_mdopt"
    device: Optional[str] = "cuda"
    preset: str = "standard"  # "quick" | "standard" | "thorough"
    run_dft: bool = False
    uma_ckpt: Optional[str] = None          # or env FAIRCHEM_UMA_CKPT
    uma_config_yml: Optional[str] = None    # or env FAIRCHEM_UMA_CONFIG
    orca_command: Optional[str] = None      # or env ORCA_COMMAND
    orca_maxcore: Optional[int] = 4000
    orca_nprocs: Optional[int] = 8
    orca_simpleinput: Optional[str] = None  # or use default

class OptimizeOut(BaseModel):
    uma: Dict[str, Any]
    dft: Optional[Dict[str, Any]] = None
    summary_path: Optional[str] = None

class PipelineIn(BaseModel):
    # generate
    claims: str
    ckpt_dir: str
    cif_dir: Optional[str] = None
    save_dir: str = "./_gens"
    use_llm: bool = True
    prototype: Optional[str] = None
    api_key: Optional[str] = None
    extra_args: Optional[List[str]] = None
    designer_client: Optional[str] = None
    # optimize
    outdir: str = "./_mdopt"
    device: Optional[str] = "cuda"
    preset: str = "standard"
    run_dft: bool = False
    uma_ckpt: Optional[str] = None
    uma_config_yml: Optional[str] = None
    orca_command: Optional[str] = None
    orca_maxcore: Optional[int] = 4000
    orca_nprocs: Optional[int] = 8
    orca_simpleinput: Optional[str] = None  # or use default


class PipelineOut(BaseModel):
    gen_path: str
    optimize: Optional[OptimizeOut] = None


# ---------- Helpers ----------
def _ensure_exists(path: str | Path, kind: str = "file/dir"):
    p = Path(path)
    if not p.exists():
        raise HTTPException(status_code=400, detail=f"{kind} not found: {path}")
    return p

def _find_newest_npz(save_dir: str | Path) -> Optional[Path]:
    p = Path(save_dir)
    cands = sorted(p.glob("*.npz"), key=lambda x: x.stat().st_mtime, reverse=True)
    return cands[0] if cands else None

def _run_gen_test(
    claims: str,
    ckpt_dir: str,
    cif_dir: Optional[str],
    save_dir: str,
    use_llm: bool,
    prototype: Optional[str],
    api_key: Optional[str],
    extra_args: Optional[List[str]],
    designer_client: Optional[str] = None,
) -> (Path, List[str], str):
    """
    Calls the README's generation entrypoint:
      - LLM scaffold:  python gen_test.py --cif-dir ... --ckpt-dir ... --prompt "<claims>" --save-dir ...
      - Offline:       python gen_test.py --no-llm --prototype rocksalt --ckpt-dir ... --save-dir ...
    Returns: (newest_npz, all_saves, log_text)
    """
    env = os.environ.copy()

    cmd: List[str] = [sys.executable, "gen_test.py", "--ckpt-dir", ckpt_dir, "--save-dir", save_dir]
    if use_llm:
        cmd += ["--prompt", claims]
        if api_key:
            cmd += ["--api-key", api_key]
        if cif_dir:
            cmd += ["--cif-dir", cif_dir]
        if designer_client:
            cmd += ["--designer_client", designer_client]
    else:
        # Offline prototype
        if not prototype:
            raise HTTPException(status_code=400, detail="prototype is required when use_llm=False.")
        cmd += ["--no-llm", "--prototype", prototype]

    if extra_args:
        cmd += list(extra_args)

    Path(save_dir).mkdir(parents=True, exist_ok=True)

    # Run
    t0 = time.time()
    try:
        out = subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT, env=env)
    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail=f"gen_test.py failed:\n{e.output}") from e

    # Resolve newest .npz produced
    newest = _find_newest_npz(save_dir)
    if newest is None:
        # Sometimes users output .npy; try a broader message
        raise HTTPException(
            status_code=500,
            detail=f"Generation finished but no .npz found in {save_dir}. Check logs:\n{out}",
        )

    files = [str(p) for p in Path(save_dir).glob("*")]
    log = f"[elapsed: {time.time()-t0:.2f}s]\n{out}"
    return newest, files, log


# ---------- Endpoints ----------
@app.get("/health")
def health():
    return {"status": "ok", "optimizer_imported": HAS_OPTIMIZER}

@app.post("/generate", response_model=GenerateOut)
def generate(inp: GenerateIn):
    _ensure_exists(inp.ckpt_dir, "ckpt_dir")
    if inp.cif_dir and not inp.use_llm:
        # allowed but unused
        pass
    if inp.use_llm:
        _ensure_exists(inp.cif_dir or "", "cif_dir")

    gen_path, files, log = _run_gen_test(
        claims=inp.claims,
        ckpt_dir=inp.ckpt_dir,
        cif_dir=inp.cif_dir,
        save_dir=inp.save_dir,
        use_llm=inp.use_llm,
        prototype=inp.prototype,
        api_key=inp.api_key,
        extra_args=inp.extra_args,
        designer_client=inp.designer_client,
    )
    return GenerateOut(gen_path=str(gen_path), saved_files=files, log=log)

@app.post("/optimize", response_model=OptimizeOut)
def optimize(inp: OptimizeIn):
    if not HAS_OPTIMIZER:
        raise HTTPException(status_code=500, detail=f"wrap_md_uma import failed: {_IMPORT_ERR}")

    _ensure_exists(inp.gen_path, "gen_path")
    Path(inp.outdir).mkdir(parents=True, exist_ok=True)

    # Allow env fallbacks
    uma_ckpt = inp.uma_ckpt or os.getenv("FAIRCHEM_UMA_CKPT")
    uma_cfg  = inp.uma_config_yml or os.getenv("FAIRCHEM_UMA_CONFIG")
    orca_cmd = inp.orca_command or os.getenv("ORCA_COMMAND")

    try:
        res = optimize_and_characterize(
            gen_path=inp.gen_path,
            outdir=inp.outdir,
            uma_ckpt=uma_ckpt,
            uma_config_yml=uma_cfg,
            device=inp.device,
            preset=inp.preset,
            run_dft=inp.run_dft,
            orca_command=orca_cmd,
            orca_maxcore=inp.orca_maxcore,
            orca_nprocs=inp.orca_nprocs,
            orca_simpleinput=inp.orca_simpleinput,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"optimize_and_characterize failed: {e}") from e

    summary = Path(inp.outdir) / "summary.json"
    return OptimizeOut(
        uma=res.get("uma", {}),
        dft=res.get("dft"),
        summary_path=str(summary) if summary.exists() else None,
    )

@app.post("/pipeline", response_model=PipelineOut)
def pipeline(inp: PipelineIn):
    # Step 1: generate
    gen_path, _, _ = _run_gen_test(
        claims=inp.claims,
        ckpt_dir=inp.ckpt_dir,
        cif_dir=inp.cif_dir,
        save_dir=inp.save_dir,
        use_llm=inp.use_llm,
        prototype=inp.prototype,
        api_key=inp.api_key,
        extra_args=inp.extra_args,
        designer_client=inp.designer_client,
    )

    # Step 2/3: optimize (optional)
    optimize_out: Optional[OptimizeOut] = None
    if HAS_OPTIMIZER:
        opt_in = OptimizeIn(
            gen_path=str(gen_path),
            outdir=inp.outdir,
            device=inp.device,
            preset=inp.preset,
            run_dft=inp.run_dft,
            uma_ckpt=inp.uma_ckpt,
            uma_config_yml=inp.uma_config_yml,
            orca_command=inp.orca_command,
            orca_maxcore=inp.orca_maxcore,
            orca_nprocs=inp.orca_nprocs,
            orca_simpleinput=inp.orca_simpleinput,
        )
        optimize_out = optimize(opt_in)  # reuse handler
    return PipelineOut(gen_path=str(gen_path), optimize=optimize_out)
