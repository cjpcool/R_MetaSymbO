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
You should run the code sequentially, in step (1) -> (2) -> (2.1) -> (3).

### 0) Environment

- **Python** ≥ 3.9  
- **PyTorch** + **PyG** (per your GeomVAE)  
- **ASE** (`pip install ase`)  
- **Fairchem / OCP** (UMA) — one of:
  - `pip install fairchem`
  - Download UMA Checkpoint from huggingface: save the downloaded "uma-s-1p1.pt" to ./checkpoint
- **ORCA** (optional, for DFT): install ORCA and ensure the binary is on `PATH` or set `ORCA_COMMAND=/path/to/orca`. Reference: https://orcaforum.kofo.mpg.de/app.php/portal

### 1) Generate a crystal (Agent-1/2 only)

**LLM scaffold mode (Agent-1):**
- **R_MetaSymbO checkpoint** on omat24dataset: download from https://drive.google.com/drive/folders/1JQ6-tAcz7B5CCfuJSiCyuYng-0eFO9GY?usp=sharing and save to ./checkpoints/omat24_rattle2

```bash
python gen_test.py \
  --no-generator \
  --designer_client 'gpt-5' \
  --api-key "" \
  --ckpt-dir ./checkpoints/omat24_rattle2 \
  --prompt "Al20Zn80 at 870K is a solid at equilibrium" \
  --save-dir ./_gens \
  --seed 42
```


Output: ./_gens/gen_union.npz (or rocksalt_union_offline.npz) with Z, cart_coords, lengths, angles.

### 2) UMA MD/relax only
- Download UMA Checkpoint from huggingface and save the downloaded "uma-s-1p1.pt" to ./checkpoint
Using **UMA (preferred)**. If you don’t pass a checkpoint, script will check `FAIRCHEM_UMA_CKPT`:
```
export FAIRCHEM_UMA_CKPT=/path/to/uma.pt
python wrap_md_uma.py \
  --gen-path ./_gens/gen_union.npz \
  --preset standard \
  --outdir ./_mdopt
  ```

**Outputs:**

`_mdopt/best.traj`, `_mdopt/best_energy.txt` and logs (when enabled).

### (2.1) Analyze optimization trajectories
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
    ,
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