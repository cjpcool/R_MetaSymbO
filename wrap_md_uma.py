#!/usr/bin/env python3
"""
UMA -> MD/Relax -> ORCA-DFT 单点表征 的一体化脚本（含可编程 API）

功能
----
1) 读取生成器输出的 .npz/.npy（atom_types, cart_coords, lengths, angles）构造 ASE Atoms
2) 使用 Fairchem (OCP) UMA 预训练模型做 MD/退火/淬火 + 0K 松弛（或 EMT 回退）
3) （可选）用 ORCA 做单点( SP + EnGrad ) 计算，解析能量、HOMO/LUMO 缝隙、偶极等
4) 将 UMA 与 DFT 结果保存到指定目录，并提供 `optimize_and_characterize(...)` 方法返回最终结果

重要说明
--------
• ORCA 为分子级非周期程序；DFT 步会在结构副本上 set_pbc(False)，并去掉晶胞，仅做单点/梯度。
• ORCA 默认输入参考你的模板并做了小幅增强：
  "M062X 6-31G* SP EnGrad D3BJ def2/J RIJCOSX TightSCF NoAutoStart MiniPrint NoPop"
• ORCA 并非所有性质都能经由 ASE 自动解析；本脚本附带轻量输出解析器提取常见量。
"""

import argparse
import os
import math
import json
import re
import numpy as np
from typing import Tuple, Dict, Any

from ase import Atoms
from ase.io import write
import torch
import shutil, subprocess


from structure_optim_modules.structure_optim import relax_0K, md_anneal, md_quench
import numpy as _np
from ase.calculators.orca import ORCA, OrcaProfile
from ase.calculators.calculator import CalculatorSetupError


# ---------------- Geometry helpers ----------------

def lengths_angles_to_cell(a: float, b: float, c: float, alpha: float, beta: float, gamma: float):
    ar, br, gr = math.radians(alpha), math.radians(beta), math.radians(gamma)
    va = np.array([a, 0.0, 0.0])
    vb = np.array([b * math.cos(gr), b * math.sin(gr), 0.0])
    cx = c * math.cos(br)
    cy = c * (math.cos(ar) - math.cos(br) * math.cos(gr)) / (math.sin(gr) + 1e-12)
    cz = math.sqrt(max(c**2 - cx**2 - cy**2, 0.0))
    vc = np.array([cx, cy, cz])
    return np.vstack([va, vb, vc])


def atoms_from_np(path: str) -> Atoms:
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    if path.endswith('.npz'):
        npz = np.load(path)
        Z = npz['atom_types'].astype(int)
        pos = npz['cart_coords'].astype(float)
        lengths = npz['lengths'].astype(float).reshape(-1)
        angles = npz['angles'].astype(float).reshape(-1)
    else:
        arr = np.load(path, allow_pickle=True)
        if isinstance(arr, np.lib.npyio.NpzFile):
            Z = arr['atom_types'].astype(int)
            pos = arr['cart_coords'].astype(float)
            lengths = arr['lengths'].astype(float).reshape(-1)
            angles = arr['angles'].astype(float).reshape(-1)
        else:
            obj = arr.item() if hasattr(arr, 'item') else arr
            Z = np.array(obj['atom_types']).astype(int)
            pos = np.array(obj['cart_coords']).astype(float)
            lengths = np.array(obj['lengths']).astype(float).reshape(-1)
            angles = np.array(obj['angles']).astype(float).reshape(-1)
    cell = lengths_angles_to_cell(*lengths.tolist(), *angles.tolist())
    return Atoms(numbers=Z.tolist(), positions=pos, cell=cell, pbc=True)

# ---------------- UMA helpers ----------------

def build_fairchem_uma(checkpoint: str, device: str = 'cuda', **kwargs):
    """Return a Fairchem/OCP UMA calculator from a checkpoint.

    Tries multiple import paths for compatibility:
        - from fairchem.core import OCPCalculator     (newer Fairchem)
        - from ocpmodels.common import OCPCalculator  (older OCP)

    Parameters
    ----------
    checkpoint : str
        Path to the UMA pre-trained model .pt/.pth checkpoint.
    device : str
        'cuda' or 'cpu'. If CUDA is unavailable, set to 'cpu'.
    kwargs : dict
        Extra args forwarded to OCPCalculator (e.g., config_yml, cpu=True, etc.).
    """

    from fairchem.core import FAIRChemCalculator
    # from fairchem.core.calculate import pretrained_mlip
    # predictor = pretrained_mlip.get_predict_unit("uma-s-1p1", device="cuda")

    # Method 2: From checkpoint
    calc = FAIRChemCalculator.from_model_checkpoint(
        name_or_path=checkpoint,
        task_name="omat",
        inference_settings="default",
        device="cuda"
    )

    return calc


def safe_optimize_with_md_loop(atoms, calc, n_loops=5, fmax=0.03, relax_cell=False,
                               anneal_kwargs=None, quench_kwargs=None, min_deltaE=1e-3,
                               outdir: str = '.'):

    atoms.calc = calc
    best = atoms.copy()
    best_E = atoms.get_potential_energy()
    print("Initial potential energy:", best_E)
    write(os.path.join(outdir, 'loop_0.traj'), atoms)
    print("Pre-relaxing to 0 K...")
    relax_0K(atoms, fmax=fmax, relax_cell=relax_cell, logfile=os.path.join(outdir, 'relax_00.log'))
    E0 = atoms.get_potential_energy()
    print("Initial potential energy after pre-relax:", E0)
    if E0 < best_E:
        best, best_E = atoms.copy(), E0
    # write(os.path.join(outdir, 'loop_1.traj'), atoms)
    print("Starting optimization loop...")
    for i in range(n_loops):
        print(f"Loop {i+1}/{n_loops}: Annealing and quenching...")
        atoms = best.copy(); atoms.calc = calc
        pos = atoms.get_positions(); atoms.set_positions(pos + _np.random.normal(0, 0.1, pos.shape))
        print(f"  Starting energy after perturbation: {atoms.get_potential_energy():.6f} eV")
        md_anneal(atoms, **(anneal_kwargs or {}))
        md_quench(atoms, **(quench_kwargs or {}))
        print("Relaxing to 0 K...")
        relax_0K(atoms, fmax=fmax, relax_cell=relax_cell, logfile=os.path.join(outdir, f'relax_{i+1:02d}.log'))
        Ei = atoms.get_potential_energy(); write(os.path.join(outdir, f'loop_{i+1}.traj'), atoms)
        
        print(f"Loop {i+1}/{n_loops}: Final energy = {Ei:.6f} eV (best so far: {best_E:.6f} eV)")

        if Ei + min_deltaE < best_E:
            best, best_E = atoms.copy(), Ei
            improvement = best_E - Ei
            print(f"  *** NEW BEST STRUCTURE! Improvement: {improvement:.6f} eV ***")
        else:
            print(f"  No improvement (ΔE = {Ei - best_E:.6f} eV)")
    print(f"\nOptimization completed. Best energy: {best_E:.6f} eV")

    return best, best_E

# ---------------- ORCA helpers ----------------


def try_orca_property_to_json(workdir: str, basename: str = "orca"):
    """
    若存在 orca_2json 且生成了 basename.property.txt，则转为 basename.property.json。
    失败时静默跳过，不影响主流程。
    """
    prop_txt = os.path.join(workdir, f"{basename}.property.txt")
    if not os.path.exists(prop_txt):
        return
    if shutil.which("orca_2json") is None:
        return
    try:
        # 等价于在终端执行：orca_2json orca_calc -property
        subprocess.run(["orca_2json", basename, "-property"], cwd=workdir,
                       check=False, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    except Exception as e:
        print("Warning: Failed to run orca_2json for property conversion.", e)


def build_orca_calculator(atoms, orca_command: str,
                          workdir: str,
                          maxcore: int = 4000,
                          nprocs: int = None,
                          orcasimpleinput: str = None,
                          extra_blocks: str = "",
                          add_default_scf=True):

    # ---- 并行与内存 ----
    if nprocs is None:
        nprocs = max(1, os.cpu_count() or 1)

    # ---- 基本关键字（可按需改）----
    if orcasimpleinput is None:
        # PBE + RIJCOSX + D3BJ：勘探/几何优化的“耐造”组合
        orcasimpleinput = "PBE def2-SVP def2/J RIJCOSX TightSCF D4 EnGrad CHELPG"

    # ---- 电荷 / 多重度（自动推断）----
    charge = None
    mult = None
    if charge is None:
        charge = 0  # 默认中性

    if mult is None:
        # 根据电子奇偶性给个“物理合理”的初始多重度
        # electron count = Z_total - charge
        Z_total = int(np.array(atoms.get_atomic_numbers()).sum())
        n_electrons = Z_total - charge
        # 偶电子 -> 单重态；奇电子 -> 双重态
        mult = 1 if (n_electrons % 2 == 0) else 2

    # ---- 若是开放壳层又没指定波函数类型，则加 UKS ----
    simple_lower = " " + orcasimpleinput.lower() + " "
    if mult > 1 and not any(x in simple_lower for x in [" uks ", " uhf ", " rohf "]):
        orcasimpleinput = "UKS " + orcasimpleinput

    # ---- Blocks 组装（顺序很重要：%maxcore 在前，后面的 block 可覆盖细节）----
    blocks_list = []
    blocks_list.append(f"%maxcore {maxcore}")

    blocks_list.append(f"""%pal
  nprocs {nprocs}
end""")

    if add_default_scf:
        # 给一份“通用耐收敛”的 SCF 设置（可按需删改）
        blocks_list.append("""%scf
  MaxIter 50
  DIISStart 3
  SOSCFStart 8
  # 金属体系有时加点电子温度有利于收敛（单位 K）
end""")
    blocks_list.append("""%output
  PrintLevel Normal
  Print[ P_OrbEn ] 2
  Print[ P_Mulliken ] 1
  Print[ P_Loewdin ] 1
  Print[ P_Mayer ] 1
  Print[ P_Hirshfeld ] 1
  Print[ P_MBIS ] 1
end""")

    blocks_list.append("""%elprop
  Dipole true
  Quadrupole true
  Polar true
end""")


    if extra_blocks:
        # 用户追加，放在最后，覆盖前述设置
        blocks_list.append(extra_blocks.strip())

    orcablocks = "\n\n".join(blocks_list) + "\n"

    # ---- 文件名/路径与 Profile ----
    os.makedirs(workdir, exist_ok=True)
    label = "orca"
    profile = OrcaProfile(command=orca_command) if orca_command else OrcaProfile()

    # ---- 构造 calculator ----
    calc = ORCA(
        profile=profile,
        directory=workdir,     # ASE ORCA 支持 directory 参数
        label=label,           # 计算文件前缀
        charge=charge,         # ★ 关键：显式设置电荷
        mult=mult,             # ★ 关键：显式设置多重度
        orcasimpleinput=orcasimpleinput,
        orcablocks=orcablocks,
    )
    return calc, os.path.join(workdir, label + '.out')


def parse_orca_output(out_file: str) -> Dict[str, Any]:
    """
    Parse ORCA output, preferring the JSON property file, with robust fallback to .out text.

    Returns keys (when available):
      - energy_hartree, energy_eV
      - dipole_vec_D (list[3]), dipole_D
      - homo_eV, lumo_eV, gap_eV           # closed-shell or best overall
      - homo_alpha_eV, lumo_alpha_eV, gap_alpha_eV
      - homo_beta_eV,  lumo_beta_eV,  gap_beta_eV
      - mulliken_charges (list[float]), mulliken_block (str)
      - S2
    """
    import os, re, json
    import numpy as np
    res: Dict[str, Any] = {}

    if not os.path.isfile(out_file):
        return res

    workdir = os.path.dirname(out_file)
    base = os.path.splitext(os.path.basename(out_file))[0]

    # ---------- helpers ----------
    def _eV(Eh: float) -> float:
        return float(Eh) * 27.211386245988

    def _try_load_json_property() -> dict | None:
        """Try several common filenames: base.property.json, base_property.json, and any '*property*.json'."""
        candidates = [
            os.path.join(workdir, base + ".property.json"),
            os.path.join(workdir, base + "_property.json"),
            os.path.join(workdir, base + ".prop.json"),
        ]
        # broaden search if not found
        if not any(os.path.isfile(p) for p in candidates):
            for fn in os.listdir(workdir):
                if fn.startswith(base) and "property" in fn.lower() and fn.lower().endswith(".json"):
                    candidates.append(os.path.join(workdir, fn))
        for p in candidates:
            if os.path.isfile(p):
                try:
                    with open(p, "r") as f:
                        return json.load(f)
                except Exception:
                    pass
        return None

    def _walk_find_props(obj, names: list[str]):
        """Search recursively for dicts like {'name': 'SCF_Energy', 'value': ...} or direct keys.
        Returns first match value or None."""
        target = {n.lower() for n in names}
        if isinstance(obj, dict):
            # direct key match
            for k, v in obj.items():
                if k.lower() in target:
                    return v
            # 'name'/'value' style
            if "name" in obj and isinstance(obj.get("name"), str) and obj["name"].lower() in target:
                return obj.get("value", None)
            # recurse
            for v in obj.values():
                ret = _walk_find_props(v, names)
                if ret is not None:
                    return ret
        elif isinstance(obj, list):
            for it in obj:
                ret = _walk_find_props(it, names)
                if ret is not None:
                    return ret
        return None

    def _parse_mulliken_block(txt: str):
        # Capture block between dashed lines after the heading
        m = re.search(r"MULLIKEN ATOMIC CHARGES[\s\S]*?\n\s*-+\s*\n([\s\S]*?)\n\s*-+", txt, re.IGNORECASE)
        charges, block = [], None
        if m:
            block = m.group(1)
            for line in block.strip().splitlines():
                # Typical line: "   1  C   -0.12345"
                mm = re.search(r"^\s*\d+\s+[A-Za-z]{1,3}\s+([-\d\.Ee+]+)", line)
                if mm:
                    try:
                        charges.append(float(mm.group(1)))
                    except Exception:
                        pass
        return charges if charges else None, block

    def _parse_S2(txt: str):
        m = re.search(r"<\s*S\^?\s*2\s*>\s*=\s*([-\d\.Ee+]+)", txt)
        return float(m.group(1)) if m else None

    def _parse_energy_from_out(txt: str):
        m = re.search(r"FINAL SINGLE POINT ENERGY\s+(-?[0-9]+\.[0-9]+)", txt, re.IGNORECASE)
        if not m:
            m = re.search(r"TOTAL SCF ENERGY\s+(-?[0-9]+\.[0-9]+)", txt, re.IGNORECASE)
        return float(m.group(1)) if m else None

    def _parse_dipole_from_out(txt: str):
        # Try vector (Debye)
        mv = re.search(
            r"Total Dipole Moment\s*\(Debye\)\s*:\s*([-\d\.Ee+]+)\s+([-\d\.Ee+]+)\s+([-\d\.Ee+]+)",
            txt, re.IGNORECASE)
        if mv:
            vec = [float(mv.group(1)), float(mv.group(2)), float(mv.group(3))]
            return vec, float(np.linalg.norm(vec))
        # Try vector (a.u.), we'll convert to Debye if needed (1 a.u. = 2.541746 D)
        mv_au = re.search(
            r"Total Dipole Moment\s*\(a\.u\.\)\s*:\s*([-\d\.Ee+]+)\s+([-\d\.Ee+]+)\s+([-\d\.Ee+]+)",
            txt, re.IGNORECASE)
        if mv_au:
            au2D = 2.541746
            vec = [float(mv_au.group(1)) * au2D,
                   float(mv_au.group(2)) * au2D,
                   float(mv_au.group(3)) * au2D]
            return vec, float(np.linalg.norm(vec))
        # Try magnitude line only
        mmag = re.search(r"Total Dipole Moment\s*:\s*([-\d\.Ee+]+)\s*Debye", txt, re.IGNORECASE)
        if mmag:
            return None, float(mmag.group(1))
        return None, None

    def _parse_mo_energies_from_out(txt: str):
        """Return dict with alpha/beta lists of occupied/unoccupied energies (eV), and gaps."""
        out = {}  # keys: alpha_occ, alpha_virt, beta_occ, beta_virt, gaps...
        # helper to parse one section into (occ_list_eV, virt_list_eV)
        def parse_section(section_text: str):
            occ_e, virt_e = [], []
            # ORCA prints columns like: NO  OCC          E(Eh)        E(eV)
            for line in section_text.splitlines():
                mm = re.search(r"\bOCC\b\s*=\s*([0-9]*\.?[0-9]+).*?E\(eV\)\s*=\s*([-\d\.Ee+]+)", line)
                if mm:
                    occ = float(mm.group(1))
                    e_ev = float(mm.group(2))
                    if occ > 1e-3:
                        occ_e.append(e_ev)
                    else:
                        virt_e.append(e_ev)
                    continue
                # older formats: columns separated by spaces
                mm2 = re.search(r"^\s*\d+\s+([0-9]*\.?[0-9]+)\s+[-\d\.Ee+]+\s+([-\d\.Ee+]+)\s*$", line)
                if mm2:
                    occ = float(mm2.group(1))
                    e_ev = float(mm2.group(2))
                    if occ > 1e-3:
                        occ_e.append(e_ev)
                    else:
                        virt_e.append(e_ev)
            occ_e.sort()
            virt_e.sort()
            return occ_e, virt_e

        def gap_from_lists(occ_e, virt_e):
            if occ_e and virt_e:
                return float(virt_e[0] - occ_e[-1]), float(occ_e[-1]), float(virt_e[0])
            return None, (occ_e[-1] if occ_e else None), (virt_e[0] if virt_e else None)

        # Find sections
        # Alpha
        ma = re.search(r"ORBITAL ENERGIES\s*\(ALPHA\)([\s\S]*?)(?:\n\s*\n|\Z)", txt, re.IGNORECASE)
        # Beta
        mb = re.search(r"ORBITAL ENERGIES\s*\(BETA\)([\s\S]*?)(?:\n\s*\n|\Z)", txt, re.IGNORECASE)
        # Closed shell (no spin tag)
        mc = re.search(r"^\s*ORBITAL ENERGIES\s*\n([\s\S]*?)(?:\n\s*\n|\Z)", txt, re.IGNORECASE | re.MULTILINE)

        gaps = []
        if ma:
            ao, av = parse_section(ma.group(1))
            g, homo, lumo = gap_from_lists(ao, av)
            out["alpha_occ"], out["alpha_virt"] = ao, av
            if homo is not None: out["homo_alpha_eV"] = float(homo)
            if lumo is not None: out["lumo_alpha_eV"] = float(lumo)
            if g is not None:
                out["gap_alpha_eV"] = float(g); gaps.append(g)
        if mb:
            bo, bv = parse_section(mb.group(1))
            g, homo, lumo = gap_from_lists(bo, bv)
            out["beta_occ"], out["beta_virt"] = bo, bv
            if homo is not None: out["homo_beta_eV"] = float(homo)
            if lumo is not None: out["lumo_beta_eV"] = float(lumo)
            if g is not None:
                out["gap_beta_eV"] = float(g); gaps.append(g)
        if (not ma) and (not mb) and mc:
            co, cv = parse_section(mc.group(1))
            g, homo, lumo = gap_from_lists(co, cv)
            if homo is not None: out["homo_eV"] = float(homo)
            if lumo is not None: out["lumo_eV"] = float(lumo)
            if g is not None:
                out["gap_eV"] = float(g); gaps.append(g)
        # overall gap as the min of available spin-resolved gaps
        if gaps and "gap_eV" not in out:
            out["gap_eV"] = float(min(gaps))
        return out

    # ---------- 1) Try JSON property file ----------
    data = _try_load_json_property()
    if data is not None:
        # Energy
        Eh = None
        for keyset in [["SCF_Energy", "SCF_ENERGY", "TOTAL_SCF_ENERGY", "SCF ENERGY"]]:
            Eh = _walk_find_props(data, keyset)
            if Eh is not None:
                break
        if isinstance(Eh, (int, float)):
            res["energy_hartree"] = float(Eh)
            res["energy_eV"] = _eV(Eh)

        # Dipole vector (Debye) and magnitude
        dip = _walk_find_props(data, ["DIPOLETOTAL", "TOTAL_DIPOLE", "DIPOLE_TOTAL", "DipoleTotal"])
        if isinstance(dip, (list, tuple)) and len(dip) >= 3:
            vec = [float(dip[0][0]), float(dip[1][0]), float(dip[2][0])]
            res["dipole_vec_D"] = vec
            res["dipole_D"] = float(np.linalg.norm(vec))
        else:
            # Some JSON variants store components separately; try to stitch
            dx = _walk_find_props(data, ["DIPOLE_X", "DipoleX"])
            dy = _walk_find_props(data, ["DIPOLE_Y", "DipoleY"])
            dz = _walk_find_props(data, ["DIPOLE_Z", "DipoleZ"])
            if all(isinstance(v, (int, float)) for v in [dx, dy, dz]):
                vec = [float(dx), float(dy), float(dz)]
                res["dipole_vec_D"] = vec
                res["dipole_D"] = float(np.linalg.norm(vec))

    # ---------- 2) Fallback: parse the .out text ----------
    with open(out_file, "r", errors="ignore") as f:
        txt = f.read()

    if "energy_eV" not in res:
        Eh = _parse_energy_from_out(txt)
        if Eh is not None:
            res["energy_hartree"] = float(Eh)
            res["energy_eV"] = _eV(Eh)

    # Dipole
    if "dipole_D" not in res:
        vec, mag = _parse_dipole_from_out(txt)
        if vec is not None:
            res["dipole_vec_D"] = [float(x) for x in vec]
            res["dipole_D"] = float(mag)
        elif mag is not None:
            res["dipole_D"] = float(mag)

    # MO energies / HOMO-LUMO gaps
    mo = _parse_mo_energies_from_out(txt)
    res.update({k: v for k, v in mo.items() if v is not None})

    # Mulliken charges
    charges, blk = _parse_mulliken_block(txt)
    if charges is not None:
        res["mulliken_charges"] = [float(x) for x in charges]
    if blk:
        res["mulliken_block"] = blk

    # <S^2>
    s2 = _parse_S2(txt)
    if s2 is not None:
        res["S2"] = float(s2)

    return res


def run_orca_dft(atoms: Atoms,
                 orca_command: str,
                 workdir: str,
                 maxcore: int = 4000,
                 nprocs: int = None,
                 orcasimpleinput: str = None,
                 extra_blocks: str = "") -> Dict[str, Any]:
    os.makedirs(workdir, exist_ok=True)
    mol = atoms.copy(); mol.set_pbc(False); mol.set_cell([0,0,0])
    calc, out_path = build_orca_calculator(atoms, orca_command, workdir, maxcore, nprocs, orcasimpleinput, extra_blocks)
    mol.calc = calc
    try:
        e = mol.get_potential_energy(); f = mol.get_forces()
    except CalculatorSetupError as e:
        raise RuntimeError(f"ORCA calculation failed to start: {e}")

    try_orca_property_to_json(workdir, basename="orca")

    write(os.path.join(workdir, 'orca_input.xyz'), mol)
    np.save(os.path.join(workdir, 'forces.npy'), f)
    parsed = parse_orca_output(out_path); parsed.setdefault('energy_eV', float(e)); parsed['forces'] = f.tolist()
    return parsed

# ---------------- Orchestrator ----------------

def optimize_and_characterize(gen_path: str,
                              outdir: str = './_mdopt',
                              uma_ckpt: str = None,
                              uma_config_yml: str = None,
                              device: str = None,
                              fallback_emt: bool = False,
                              preset: str = 'standard',
                              loops: int = 5,
                              fmax: float = 0.03,
                              relax_cell: bool = False,
                              anneal_defaults: Dict[str, Any] = None,
                              quench_defaults: Dict[str, Any] = None,
                              run_dft: bool = True,
                              orca_command: str = 'orca',
                              orca_maxcore: int = 4000,
                              orca_nprocs: int = None,
                              orca_simpleinput: str = None,
                              orca_extra_blocks: str = "") -> Dict[str, Any]:
    # auto device
    if device is None:
        try:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        except Exception:
            device = 'cpu'
    os.makedirs(outdir, exist_ok=True)

    atoms = atoms_from_np(gen_path)

    # UMA calc
    calc = None
    if uma_ckpt is None:
        uma_ckpt = os.getenv('FAIRCHEM_UMA_CKPT')
    if uma_ckpt is not None:
        extra = {}
        if uma_config_yml is None:
            uma_config_yml = os.getenv('FAIRCHEM_UMA_CONFIG')
        if uma_config_yml: extra['config_yml'] = uma_config_yml
        try:
            calc = build_fairchem_uma(uma_ckpt, device=device, **extra)
        except Exception:
            if not fallback_emt:
                raise
    if calc is None:
        if fallback_emt:
            from ase.calculators.emt import EMT; calc = EMT()
        else:
            raise RuntimeError('No calculator available. Provide UMA checkpoint or enable fallback_emt.')

    # MD presets
    npt=False; pressure_atm=1.0; tstep_fs=2.0; friction=0.01
    T_start, T_peak, T_hold_ps = 300.0, 900.0, 10.0
    T_final, cool_ps = 100.0, 20.0
    if preset=='quick':
        loops=min(loops,2); T_start, T_peak, T_hold_ps = 300.0, 600.0, 5.0; T_final, cool_ps, tstep_fs = 150.0, 10.0, 2.0; fmax=max(fmax,0.05)
    elif preset=='thorough':
        loops=max(loops,8); T_start, T_peak, T_hold_ps = 300.0, 1400.0, 30.0; T_final, cool_ps, tstep_fs = 50.0, 60.0, 1.0; fmax=min(fmax,0.02)
    anneal_kwargs=dict(T_start=T_start, T_peak=T_peak, T_hold_ps=T_hold_ps, npt=npt, pressure_atm=pressure_atm, tstep_fs=tstep_fs, friction=friction)
    quench_kwargs=dict(T_final=T_final, cool_ps=cool_ps, npt=npt, pressure_atm=pressure_atm, tstep_fs=tstep_fs, friction=friction)
    if anneal_defaults: anneal_kwargs.update(anneal_defaults)
    if quench_defaults: quench_kwargs.update(quench_defaults)

    # UMA optimize (with safe fallback)
    # best_atoms, best_E = safe_optimize_with_md_loop(atoms, calc, n_loops=loops, fmax=fmax, relax_cell=relax_cell,
    #                                                     anneal_kwargs=anneal_kwargs, quench_kwargs=quench_kwargs, min_deltaE=1e-3, outdir=outdir)
    best_atoms = atoms
    atoms.calc = calc
    best_E = atoms.get_potential_energy()
    print('atom pos', atoms.positions)
    atoms.set_cell(torch.tensor([[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]]))
    
    # Save UMA results
    write(os.path.join(outdir, 'best.traj'), best_atoms)
    write(os.path.join(outdir, 'best.xyz'), best_atoms)
    with open(os.path.join(outdir, 'best_energy.txt'), 'w') as f: f.write(f"{best_E:.6f}\n")

    result: Dict[str, Any] = {'uma': {'energy_eV': float(best_E), 'traj': os.path.join(outdir, 'best.traj'), 'xyz': os.path.join(outdir, 'best.xyz')}}
    print(f"UMA optimization done. results:", result)
    # ORCA
    if run_dft:
        print("Starting ORCA DFT single-point calculation...")
        orca_dir = os.path.join(outdir, 'orca_sp')
        dft = run_orca_dft(best_atoms, orca_command=orca_command, workdir=orca_dir, maxcore=orca_maxcore,
                           nprocs=orca_nprocs, orcasimpleinput=orca_simpleinput, extra_blocks=orca_extra_blocks)
        with open(os.path.join(orca_dir, 'dft_results.json'), 'w') as f: json.dump(dft, f, indent=2)
        result['dft'] = dft
        print("ORCA DFT done. results:", dft)
    return result

# ---------------- CLI ----------------

def main():
    try:
        _AUTO_DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    except Exception:
        _AUTO_DEVICE = 'cpu'

    ap = argparse.ArgumentParser(description='UMA optimization and ORCA single-shot DFT results extraction script.')
    ap.add_argument('--gen-path', required=True, help='generated .npz/.npy path')
    ap.add_argument('--outdir', default='./_mdopt', help='Output dir')

    # UMA
    ap.add_argument('--ckpt', default=os.getenv('FAIRCHEM_UMA_CKPT'), help='UMA checkpoint path, default env FAIRCHEM_UMA_CKPT')
    ap.add_argument('--config-yml', default=os.getenv('FAIRCHEM_UMA_CONFIG'), help='UMA config YAML, optional, default env FAIRCHEM_UMA_CONFIG')
    ap.add_argument('--device', default=_AUTO_DEVICE)
    ap.add_argument('--fallback-emt', action='store_true')
    ap.add_argument('--loops', type=int, default=5)
    ap.add_argument('--fmax', type=float, default=0.03)
    ap.add_argument('--relax-cell', action='store_true')
    ap.add_argument('--preset', type=str, default='standard', choices=['quick','standard','thorough'])

    # ORCA
    ap.add_argument('--run-dft', action='store_true')
    ap.add_argument('--orca-command', default=os.getenv('ORCA_COMMAND', 'orca'))
    ap.add_argument('--nprocs', type=int, default=max(1, os.cpu_count() or 1))
    ap.add_argument('--maxcore', type=int, default=4000)
    ap.add_argument('--orcasimpleinput', default=None)
    ap.add_argument('--orcablocks-extra', default='')

    args = ap.parse_args()

    res = optimize_and_characterize(
        gen_path=args.gen_path,
        outdir=args.outdir,
        uma_ckpt=args.ckpt,
        uma_config_yml=args.config_yml,
        device=args.device,
        fallback_emt=args.fallback_emt,
        preset=args.preset,
        loops=args.loops,
        fmax=args.fmax,
        relax_cell=args.relax_cell,
        run_dft=args.run_dft,
        orca_command=args.orca_command,
        orca_maxcore=args.maxcore,
        orca_nprocs=args.nprocs,
        orca_simpleinput=args.orcasimpleinput,
        orca_extra_blocks=args.orcablocks_extra,
    )

    summary = {
        'uma_energy_eV': res['uma']['energy_eV'],
        'uma_traj': res['uma']['traj'],
        'dft_energy_eV': res.get('dft', {}).get('energy_eV'),
        'dft_gap_eV': res.get('dft', {}).get('gap_eV'),
        'dft_dipole_D': res.get('dft', {}).get('dipole_D'),
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    with open(os.path.join(args.outdir, 'summary.json'), 'w') as f:
        json.dump(res, f, indent=2, ensure_ascii=False)

if __name__ == '__main__':
    main()
