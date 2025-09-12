#!/usr/bin/env python
import os
import argparse
from pathlib import Path

import torch
from tqdm import tqdm

from datasets import LatticeModulus
from model.modal_agents   import MetamatGenAgents, MAX_NODE_NUM


def parse_args() -> argparse.Namespace:
    """Read hyper‑parameters from the shell."""
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # high‑level options
    parser.add_argument("--prompt", required=True, type=str,
                        default="Design a structure with high stiffness, with less nodes.",
                        help="Natural‑language prompt handed to the agent")
    parser.add_argument('--api_key', required=True, type=str, default="",
                        help="API key for the client models (gpt api key)")

    # compute / storage
    parser.add_argument("--cuda", type=int, default=0,
                        help="CUDA GPU id; set to -1 for CPU")
    parser.add_argument("--root", type=Path,
                        default=Path(__file__).resolve().parent,
                        help="Project root used to locate checkpoints, etc.")

    # agent 1 settings
    parser.add_argument("--designer_client", type=str, default="gpt-4o-mini",
                        help="Client model for designer agent")
    # agent 3 settings
    parser.add_argument('--supervisor_client', type=str, default="gpt-4.1",
                         help="Client model for supervisor agent")
    
    # agent 2 checkpoint & model settings
    parser.add_argument("--backbone", choices=["vae", "diffusion"], default="vae",)
    parser.add_argument("--save_name",     default="disent_diff_cond_128_vae")
    parser.add_argument("--save_name_ae",  default="checkpoints/vae_cond_128_beta001_dis_same_100_frac")
    parser.add_argument("--latent_dim",    type=int, default=128)

    # data settings
    parser.add_argument("--dataset_path",  required=True, type=Path,
                        default=Path("/home/username/datasets/metamaterial/LatticeModulus"))
    
    parser.add_argument("--file_name",     default="data")
    parser.add_argument("--select_max_node_num", type=int, default=30)
    parser.add_argument("--train_size",    type=int, default=8000)
    parser.add_argument("--valid_size",    type=int, default=1)
    parser.add_argument("--seed",          type=int, default=42)

    # agent collaboration / diffusion arguments
    parser.add_argument("--max_collaboration_num", type=int, default=5, help="Maximum trial times for collaboration loops")
    parser.add_argument("--max_evaluate_num", type=int, default=5, help="Maximum collaboration times of collaboration of Agent 13 and 23")
    parser.add_argument("--evaluation_threshold", type=float, default=0.6, help="Threshold for evaluation")
    parser.add_argument("--logic_mode",     choices=["union", "mix", "int", "neg"], default="union")
    parser.add_argument("--fusion_thresh",  type=float, default=0.5)
    parser.add_argument("--mix_lam",        type=float, default=0.3)
    parser.add_argument("--num_steps_lattice", type=int, default=300)
    parser.add_argument("--num_step_geo",   type=int, default=50)
    parser.add_argument("--condition_vec",   type=str, default='None', help="Condition vector for the model, a list of float, dim=12")
    parser.add_argument("--optimize_network_params", action="store_true",
                        help="Whether to fine‑tune network weights during latent otpimization.")
    parser.add_argument('--save_dir', type=str, default='results/lattices', help='Directory to save the generation results')
    parser.add_argument("--verbose",        action="store_true")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # ── device ──────────────────────────────────────────────────────────────────
    if args.cuda >= 0 and torch.cuda.is_available():
        device = torch.device(f"cuda:{args.cuda}")
    else:
        device = torch.device("cpu")
    print("Using device:", device)


    condition_vec = eval(args.condition_vec)
    if condition_vec is not None:
        condition_vec = torch.tensor(condition_vec, dtype=torch.float32).to(device).view(1, -1)

    # ── dataset ─────────────────────────────────────────────────────────────────
    dataset = LatticeModulus(args.dataset_path, file_name=args.file_name)
    indices  = [
        i for i, data in enumerate(tqdm(dataset, desc="Filtering dataset"))
        if data.num_atoms <= MAX_NODE_NUM
        and data.num_edges <= MAX_NODE_NUM * 2
    ]
    dataset  = dataset[indices]
    split    = dataset.get_idx_split(len(dataset),
                                     train_size=args.train_size,
                                     valid_size=args.valid_size,
                                     seed=args.seed)
    train_dataset = dataset[split["train"]]
    valid_dataset = dataset[split["valid"]]
    test_dataset  = dataset[split["test"]]
    print(f"Dataset sizes ➜ train {len(train_dataset)}, valid {len(valid_dataset)}, test {len(test_dataset)}")


    assert condition_vec is None or (dataset[0].y.shape[-1] == condition_vec.shape[-1]), \
        f"Condition vector shape {condition_vec.shape} does not match dataset y shape {dataset[0].y.shape}"

    # ── initialise agent ────────────────────────────────────────────────────────
    agents = MetamatGenAgents(
        root=args.root,
        ckpt_dir=args.save_name_ae,
        device=device,
        backbone="vae",
        latent_dim=args.latent_dim,
        designer_client=args.designer_client,
        supervisor_client=args.supervisor_client,
        api_key=args.api_key,
        evaluation_threshold=args.evaluation_threshold,
        max_evaluate_num=args.max_evaluate_num,
    )

    # ── run agent ───────────────────────────────────────────────────────────────
    agents.collaborative_end_to_end_generation(
        dataset,
        args.prompt,
        logic_mode=args.logic_mode,
        num_steps_lattice=args.num_steps_lattice,
        num_steps_geo=args.num_step_geo,
        max_collaboration_num=args.max_collaboration_num,
        optimize_network_params=args.optimize_network_params,
        fusion_thresh=args.fusion_thresh,
        mix_lam=args.mix_lam,
        condition=condition_vec,
        select_max_node_num=args.select_max_node_num,
        verbose=args.verbose,
        save_dir=args.save_dir,
    )


if __name__ == "__main__":
    main()
