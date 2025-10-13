"""
Crystal generation smoke test (Agent‑1/2 only; no Supervisor).

This script:
  1) Loads an ASE-backed dataset and wraps it with OMAT24Dataset (cartesian+fractional, Z).
  2) Instantiates CrystalGenAgents (your revised modal_agents.py).
  3) (Optional) Primes the lattice normalizer with a small dataset sample.
  4) Generates a crystal structure via agent12_generation() and saves an .npz.

It supports two scaffold modes:
  • LLM mode (default): uses Agent‑1 translator (requires OpenAI/Gemini API key).
  • Offline mode (–-offline-scaffold): uses a built-in prototype (e.g., NaCl rocksalt), bypassing LLM.

Outputs (.npz):
  atom_types (Z), cart_coords (Å), lengths [a,b,c] (Å), angles [α,β,γ] (deg), prop_list (if any)
"""

import argparse
import os
import sys
import random
from typing import List, Tuple

import numpy as np
import torch
from torch_geometric.loader import DataLoader
from torch_geometric.nn import radius_graph

from torch_geometric.data import Data, Batch

# --- project imports ---
try:
    from datasets import OMAT24Dataset
except Exception as e:
    print("[WARN] Failed to import OMAT24Dataset. Make sure dataset_omat24.py is on PYTHONPATH.")
    raise

try:
    from model.modal_agents import CrystalGenAgents
except Exception as e:
    print("[WARN] Failed to import CrystalGenAgents. Make sure your revised modal_agents.py is on PYTHONPATH.")
    raise

# Optional ASE helpers
try:
    from ase.io import read
    import ase.db
except Exception as e:
    read = None
    ase = None


def build_cif_dir_dataset(cif_dir: str):
    """Very small ASE-like dataset wrapper from a directory of CIFs."""
    assert read is not None, "ASE not installed. Please `pip install ase`."
    files = [os.path.join(cif_dir, f) for f in os.listdir(cif_dir) if f.lower().endswith('.cif')]
    files.sort()
    class _DIR:
        def __init__(self, files):
            self.files = files
        def __len__(self):
            return len(self.files)
        def get_atoms(self, idx):
            return read(self.files[idx])
    return _DIR(files)


def build_ase_db_dataset(db_path: str):
    """Very small ASE-like dataset wrapper from an ASE .db file (rows with atoms)."""
    assert ase is not None, "ASE not installed. Please `pip install ase`."
    db = ase.db.connect(db_path)
    ids = [row.id for row in db.select()]
    class _DB:
        def __init__(self, db, ids):
            self.db = db
            self.ids = ids
        def __len__(self):
            return len(self.ids)
        def get_atoms(self, idx):
            return self.db.get_atoms(id=self.ids[idx])
    return _DB(db, ids)


# ---------- Offline scaffold prototypes (cartesian) ----------
# Diamond (Si), Rocksalt (NaCl); units Å
import math

def lengths_angles_to_cell(a: float, b: float, c: float, alpha: float, beta: float, gamma: float) -> np.ndarray:
    """Return 3x3 lattice vectors from lengths (Å) and angles (deg)."""
    # convert to radians
    ar, br, gr = math.radians(alpha), math.radians(beta), math.radians(gamma)
    va = np.array([a, 0.0, 0.0])
    vb = np.array([b * math.cos(gr), b * math.sin(gr), 0.0])
    cx = c * math.cos(br)
    cy = c * (math.cos(ar) - math.cos(br) * math.cos(gr)) / (math.sin(gr) + 1e-12)
    cz = math.sqrt(max(c**2 - cx**2 - cy**2, 0.0))
    vc = np.array([cx, cy, cz])
    return np.vstack([va, vb, vc])  # shape (3,3)


def frac_to_cart(frac: np.ndarray, lengths: np.ndarray, angles: np.ndarray) -> np.ndarray:
    cell = lengths_angles_to_cell(*lengths.tolist(), *angles.tolist())
    return frac @ cell


def offline_scaffold(prototype: str = 'rocksalt') -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (z, cart_coords, batch, lengths, angles, num_atoms) for a tiny cell."""
    if prototype.lower() == 'diamond':
        # Si diamond: a=5.43; two atoms per primitive cell
        lengths = torch.tensor([[5.43, 5.43, 5.43]], dtype=torch.float)
        angles  = torch.tensor([[90.0, 90.0, 90.0]], dtype=torch.float)
        frac = np.array([
            [0.0, 0.0, 0.0],
            [0.25, 0.25, 0.25],
        ], dtype=float)
        cart = frac_to_cart(frac, lengths[0].numpy(), angles[0].numpy())
        z = torch.tensor([14, 14], dtype=torch.long)
    else:
        # Rocksalt NaCl: a=5.64; two atoms per primitive cell
        lengths = torch.tensor([[5.64, 5.64, 5.64]], dtype=torch.float)
        angles  = torch.tensor([[90.0, 90.0, 90.0]], dtype=torch.float)
        frac = np.array([
            [0.0, 0.0, 0.0],      # Na
            [0.5, 0.5, 0.5],      # Cl
        ], dtype=float)
        cart = frac_to_cart(frac, lengths[0].numpy(), angles[0].numpy())
        z = torch.tensor([11, 17], dtype=torch.long)

    cart_coords = torch.tensor(cart, dtype=torch.float)
    batch = torch.zeros(len(z), dtype=torch.long).view(-1)  # single structure
    num_atoms = torch.tensor([len(z)], dtype=torch.long)
    return z, cart_coords, batch, lengths, angles, num_atoms


# ---------------- Timing Decorator ----------------
def timed(func=None):
    """函数计时装饰器。

    使用方式: 
    1) 直接装饰: @timed  -> 调用时 label 默认为函数名。
    2) 动态包装: wrapped = timed(original_fn); wrapped(..., timer_label='自定义标签', save_dir='path')

    调用时可额外传入:
        timer_label: 自定义计时块名称
        save_dir: 若提供, 记录到 save_dir/generation_time.txt

    返回值: (result, elapsed_seconds)
    """
    import functools, time

    if func is None:
        # 支持 @timed() 形式（目前不需要参数，保留扩展点）
        def _wrap(f):
            return timed(f)
        return _wrap

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        label = kwargs.pop('timer_label', None) or func.__name__
        save_dir = kwargs.pop('save_dir', None)
        start = time.time()
        print(f"[TIMER] {label} started.")
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        print(f"[TIMER] {label} finished in {elapsed:.3f} s.")
        if save_dir is not None:
            try:
                os.makedirs(save_dir, exist_ok=True)
                with open(os.path.join(save_dir, 'generation_time.txt'), 'a', encoding='utf-8') as f:
                    f.write(f"{label}: {elapsed:.6f} seconds\n")
            except Exception as e:
                print(f"[TIMER][WARN] Failed to write timing log: {e}")
        return result, elapsed

    return wrapper

# 兼容旧接口: 仍保留 run_with_timer 以免其它脚本引用
def run_with_timer(label: str, fn, *args, save_dir: str = None, **kwargs):
    wrapped = timed(fn)
    return wrapped(*args, timer_label=label, save_dir=save_dir, **kwargs)




# ---------- Main ----------

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--cif-dir', type=str, default=None, help='Directory of CIFs for ASE dataset.')
    p.add_argument('--ase-db', type=str, default=None, help='ASE .db file with structures.')
    p.add_argument('--ase-omat', type=str, default=None, help='omat dataset path.')
    
    p.add_argument('--cutoff', type=float, default=5.0, help='Radius for neighbor graph inside OMAT24Dataset.')
    p.add_argument('--prompt', type=str, default='Rocksalt ionic conductor (NaCl-like).', help='Design requirement for Agent-1.')
    p.add_argument('--api-key', type=str, default='', help='Provide OpenAI/Gemini API key.')
    p.add_argument('--ckpt-dir', type=str, default='./checkpoints/omat24_rattle2', help='Where best_ae_model.pt lives.')
    p.add_argument('--save-dir', type=str, default='./_gens', help='Output folder for generated npz.')
    p.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--logic-mode', type=str, default='union', choices=['union', 'mix', 'int', 'neg'])
    p.add_argument('--no-llm', action='store_true', help='Use offline built-in scaffold (skip Agent-1 LLM).')
    p.add_argument('--no-generator', action='store_true', help='Skip Agent-2 generation.')
    p.add_argument('--prototype', type=str, default='rocksalt', choices=['rocksalt', 'diamond'], help='Offline scaffold type.')
    p.add_argument('--seed', type=int, default=1337)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # ---- Build ASE dataset wrapper
    if args.no_generator:
        ase_ds = None  # will use offline scaffold only
        print('[INFO] --no-generator flag set; skipping dataset load.')
    else:
        if args.cif_dir:
            ase_ds = build_cif_dir_dataset(args.cif_dir)
        elif args.ase_db:
            ase_ds = build_ase_db_dataset(args.ase_db)
        elif args.ase_omat:
            from fairchem.core.datasets import AseDBDataset
            ase_ds = AseDBDataset(config=dict(src=args.ase_omat))
        else:
            print('[INFO] No dataset path provided; using a tiny 1-sample offline scaffold just to run the model.\n'
                '       For realistic use, pass --cif-dir DIR or --ase-db FILE.')
            ase_ds = None

    if ase_ds is not None:
        dataset = OMAT24Dataset(ase_ds, cutoff=args.cutoff)
    else:
        # Build a minimal stub dataset with one item if needed
        class _StubOMAT24(torch.utils.data.Dataset):
            def __init__(self):
                # lengths/angles placeholders to satisfy normalizer
                self.lengths = []
                self.angles = []
                self.y = None
            def __len__(self):
                return 1
            def __getitem__(self, idx):
                from torch_geometric.data import Data
                z, cart, batch, lengths, angles, num_atoms = offline_scaffold(args.prototype)
                frac = torch.zeros_like(cart)  # unused downstream
                data = Data(
                    frac_coords=frac,
                    cart_coords=cart,
                    edge_index = radius_graph(cart, r=args.cutoff),
                    node_type=z,
                    num_atoms=num_atoms,
                    lengths=lengths,
                    angles=angles,
                    batch=torch.zeros(len(z), dtype=torch.long),
                    y=torch.zeros(1,1)
                )
                self.lengths.append(lengths)
                self.angles.append(angles)
                return data
            def collate(self, data_list):

                return Batch.from_data_list(data_list)
        dataset = _StubOMAT24()

    # ---- Build agent
    agent = CrystalGenAgents(
        root='.',
        ckpt_dir=args.ckpt_dir,
        device=args.device,
        designer_client='gpt-4o-mini',  # ignored if --no-llm
        api_key=args.api_key,
    ).to(args.device)


    os.makedirs(args.save_dir, exist_ok=True)

    # 创建计时包装的协同函数
    collab_timed = timed(agent.collaborate_between_agents_12)

    if args.no_llm:
        print('[INFO] Using offline scaffold prototype:', args.prototype)
        scaffold = offline_scaffold(args.prototype)
        batch_data = next(iter(DataLoader(dataset, batch_size=1, shuffle=True)))
        _final, elapsed = collab_timed(
            batch_data,
            scaffold,
            logic_mode=args.logic_mode,
            num_steps_lattice=30,
            num_steps_geo=300,
            optimize_network_params=False,
            mix_lam=0.2,
            condition=None,
            lam_node=1.0,
            lam_keep=1.0,
            lam_prior=1e-4,
            timer_label='Agent12 Offline Generation ' + args.prompt,
            save_dir=args.save_dir,
        )
        _, lengths_pred, angles_pred, coords_gen, z_gen = _final
        out = os.path.join(args.save_dir, f'{args.prototype}_{args.logic_mode}_offline.npz')
        np.savez(out,
                 atom_types=z_gen.detach().cpu().numpy(),
                 cart_coords=coords_gen.detach().cpu().numpy(),
                 lengths=lengths_pred.detach().cpu().view(-1).numpy(),
                 angles=angles_pred.detach().cpu().view(-1).numpy())
        print('[DONE] Saved generation to', out, f"(elapsed {elapsed:.3f}s)")
        return
    
    else:
        # LLM mode: one-shot A1→A2
        print('[INFO] LLM scaffold mode. Provide API key for your selected client.')
        obj, cond, prop, req = agent.entity_extraction(args.prompt)
        print(f"[INFO] ------------------------\n \
            Entity Extraction Results:\n \
                Object: {obj},\n Conditions: {cond},\n Properties: {prop},\n Design requirements: {req}\n \
                    ------------------------")
        
        print('[INFO] Geting scaffold from Agent-1...')
        z, cart_coords, batch, lengths, angles, num_atoms = agent.get_scaffold(req, visualize_results=False)
        scaffold = (z, cart_coords, batch, lengths, angles, num_atoms)
        # Build a Batch directly from the scaffold so batch_data matches its structure

        z_s, cart_s, batch_s, lengths_s, angles_s, num_atoms_s = scaffold
        frac_s = torch.zeros_like(cart_s)

        # Build neighbor graph on CPU, then move to device with the Batch
        edge_index = radius_graph(cart_s.cpu(), r=args.cutoff, batch=(batch_s.cpu() if batch_s is not None else None))

        if args.no_generator:
            print('[INFO] --no-generator flag set; skipping Agent-2 generation.')
            lengths_pred, angles_pred, coords_gen, z_gen = lengths_s, angles_s, cart_s, z_s
        else:
            print('[INFO] Generating scaffold with Agent-2...')
            data = Data(
                frac_coords=frac_s,
                cart_coords=cart_s,
                edge_index=edge_index,
                node_type=z_s,
                num_atoms=num_atoms_s,
                lengths=lengths_s,
                angles=angles_s,
                batch=torch.zeros(len(z_s), dtype=torch.long),
                y=torch.zeros(1, 1),
            )
            batch_data = Batch.from_data_list([data]).to(args.device)
            _final, elapsed = collab_timed(
                batch_data,
                scaffold,
                logic_mode=args.logic_mode,
                num_steps_lattice=30,
                num_steps_geo=300,
                optimize_network_params=False,
                mix_lam=0.2,
                condition=None,
                lam_node=1.0,
                lam_keep=1.0,
                lam_prior=1e-4,
                timer_label='Agent12 LLM Generation ' + args.prompt,
                save_dir=args.save_dir,
            )
            _, lengths_pred, angles_pred, coords_gen, z_gen = _final
            
        out = os.path.join(args.save_dir, f'gen_{args.logic_mode}.npz')
        np.savez(out,
                atom_types=z_gen.detach().cpu().numpy(),
                cart_coords=coords_gen.detach().cpu().numpy(),
                lengths=lengths_pred.detach().cpu().view(-1).numpy(),
                angles=angles_pred.detach().cpu().view(-1).numpy())
        print('[DONE] Saved generation to', out, f"(elapsed {elapsed:.3f}s)")


if __name__ == '__main__':
    main()
