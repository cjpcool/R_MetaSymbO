#!/usr/bin/env python3
"""
Enhanced analysis script for structure optimization results (CLI-ready).

Features
--------
• Point it at any folder / filename pattern (e.g., _mdopt/*.traj, loop_*.traj)
• Tune thresholds for "stuck" energy and "small-change" RMSD
• Save plots to files (PNG/SVG/PDF) and/or suppress GUI (headless runs)
• Export energies/volumes/RMSDs to JSON/CSV
• Optional side-by-side compare: initial (loop_0.traj) vs final/best

Examples
--------
# Basic (analyze current dir loop_*.traj, show plots)
python analyze_optimization.py

# Analyze another folder and save plots/JSON
python analyze_optimization.py --root ./_mdopt --pattern "loop_*.traj" \
  --save-dir ./_mdopt/analysis --export-json metrics.json --no-show

# Tighter thresholds, SVG plots, and do a structure compare
python analyze_optimization.py --root ./_mdopt --img-format svg --compare
"""
import os
import glob
import json
import csv
import argparse
import warnings
from typing import List, Tuple, Dict, Any

import numpy as np
import matplotlib.pyplot as plt
from ase.io import read
from ase.visualize.plot import plot_atoms
from ase.data.colors import jmol_colors 
import matplotlib.patches as mpatches

warnings.filterwarnings("ignore")


def _safe_energy(atoms):
    """Try to get energy from a traj; fall back to info dict if calculator missing."""
    try:
        return float(atoms.get_potential_energy())
    except Exception:
        for key in ("energy", "free_energy", "E", "potential_energy"):
            if key in atoms.info:
                try:
                    return float(atoms.info[key])
                except Exception:
                    pass
    return float("nan")


def find_traj_files(root: str, pattern: str) -> List[str]:
    root = os.path.abspath(root)
    return sorted([f for f in glob.glob(os.path.join(root, pattern)) if os.path.isfile(f)])


def analyze_optimization_progress(
    root: str = ".",
    pattern: str = "loop_*.traj",
    stuck_threshold: float = 1e-6,
    small_change_threshold: float = 0.1,
    save_dir: str = None,
    show: bool = True,
    img_format: str = "png",
    dpi: int = 150,
) -> Tuple[List[float], List[float], List[Any]]:
    """Analyze the optimization progress from trajectory files."""
    traj_files = find_traj_files(root, pattern)
    if not traj_files:
        print(f"No trajectory files found under {root!r} matching {pattern!r}")
        return [], [], []

    print(f"Found {len(traj_files)} trajectory files")

    energies: List[float] = []
    loop_numbers: List[int] = []
    volumes: List[float] = []
    structures: List[Any] = []

    for traj_file in traj_files:
        try:
            atoms = read(traj_file)
            energy = _safe_energy(atoms)
            volume = atoms.get_volume()

            # Extract loop number from filename '.../loop_<N>.traj'
            base = os.path.basename(traj_file)
            loop_num = int(base.split("_")[1].split(".")[0])

            energies.append(energy)
            loop_numbers.append(loop_num)
            volumes.append(volume)
            structures.append(atoms)

            print(f"Loop {loop_num:3d}: E = {energy: .6f} eV, V = {volume:.3f} Å³")
        except Exception as e:
            print(f"Error reading {traj_file}: {e}")

    if not energies:
        return [], [], []

    # ---- Plots ----
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

    # Energy vs loop
    ax1.plot(loop_numbers, energies, "o-", linewidth=2, markersize=6)
    ax1.set_xlabel("Optimization Loop")
    ax1.set_ylabel("Potential Energy (eV)")
    ax1.set_title("Energy Evolution During Optimization")
    ax1.grid(True, alpha=0.3)

    # Highlight if energy gets stuck
    if len(energies) > 1:
        energy_changes = np.diff(energies)
        stuck_loops = np.where(np.abs(energy_changes) < stuck_threshold)[0]
        if len(stuck_loops) > 0:
            ax1.axhline(
                y=energies[stuck_loops[0] + 1],
                color="red",
                linestyle="--",
                alpha=0.7,
                label=f"Stuck @ {energies[stuck_loops[0] + 1]:.6f} eV",
            )
            ax1.legend()

    # Volume vs loop
    ax2.plot(loop_numbers, volumes, "s-", linewidth=2, markersize=6)
    ax2.set_xlabel("Optimization Loop")
    ax2.set_ylabel("Cell Volume (Å³)")
    ax2.set_title("Volume Evolution")
    ax2.grid(True, alpha=0.3)

    # Energy change per loop
    if len(energies) > 1:
        ax3.bar(loop_numbers[1:], np.diff(energies), alpha=0.7)
        ax3.set_xlabel("Optimization Loop")
        ax3.set_ylabel("Energy Change (eV)")
        ax3.set_title("Energy Change Per Loop")
        ax3.axhline(y=0, color="black", linestyle="-", alpha=0.3)
        ax3.grid(True, alpha=0.3)

    # Energy histogram to show if stuck in local minimum
    ax4.hist(energies, bins=min(10, len(energies)), alpha=0.7, edgecolor="black")
    ax4.set_xlabel("Potential Energy (eV)")
    ax4.set_ylabel("Frequency")
    ax4.set_title("Energy Distribution")
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        outpath = os.path.join(save_dir, f"optimization_summary.{img_format}")
        plt.savefig(outpath, dpi=dpi, bbox_inches="tight")
        print(f"[Saved] {outpath}")
        if not show:
            plt.close(fig)

    if show:
        plt.show()

    # Analyze structural changes (RMSD)
    analyze_structural_changes(
        structures, loop_numbers, small_change_threshold, save_dir, show, img_format, dpi
    )

    return energies, volumes, structures


def analyze_structural_changes(
    structures: List[Any],
    loop_numbers: List[int],
    small_change_threshold: float = 0.1,
    save_dir: str = None,
    show: bool = True,
    img_format: str = "png",
    dpi: int = 150,
) -> List[float]:
    """Analyze how much the structure changes between loops."""
    if len(structures) < 2:
        return []

    print("\n" + "=" * 60)
    print("STRUCTURAL CHANGE ANALYSIS")
    print("=" * 60)

    rmsds = []
    for i in range(1, len(structures)):
        pos1 = structures[i - 1].get_positions()
        pos2 = structures[i].get_positions()

        # Simple RMSD (assumes same ordering)
        rmsd = float(np.sqrt(np.mean((pos1 - pos2) ** 2)))
        rmsds.append(rmsd)

        print(f"Loop {loop_numbers[i-1]:3d} → {loop_numbers[i]:3d}: RMSD = {rmsd:.4f} Å")

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(loop_numbers[1:], rmsds, "o-", linewidth=2, markersize=6, color="red")
    ax.set_xlabel("Optimization Loop")
    ax.set_ylabel("RMSD from Previous Structure (Å)")
    ax.set_title("Structural Changes Between Loops")
    ax.grid(True, alpha=0.3)

    if len(rmsds) > 0:
        ax.axhline(
            y=small_change_threshold,
            color="orange",
            linestyle="--",
            alpha=0.7,
            label=f"Small-change threshold ({small_change_threshold:.3f} Å)",
        )
        ax.legend()

    plt.tight_layout()

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        outpath = os.path.join(save_dir, f"structural_changes.{img_format}")
        plt.savefig(outpath, dpi=dpi, bbox_inches="tight")
        print(f"[Saved] {outpath}")
        if not show:
            plt.close(fig)

    if show:
        plt.show()

    return rmsds


def compare_initial_vs_final(
    root: str = ".",
    pattern: str = "loop_*.traj",
    initial_file: str = "loop_0.traj",
    best_file: str = "best.traj",
    save_dir: str = None,
    show: bool = True,
    img_format: str = "png",
    dpi: int = 150,
):
    """Compare initial and final structures (highest loop and optional best.traj)."""
    try:
        initial_path = os.path.join(root, initial_file)
        initial = read(initial_path)

        final_files = [
            f for f in find_traj_files(root, pattern) if os.path.basename(f) != initial_file
        ]
        if not final_files:
            print("No final structures found")
            return

        # Highest numbered loop
        loop_nums = [int(os.path.basename(f).split("_")[1].split(".")[0]) for f in final_files]
        final_loop = max(loop_nums)
        final_path = os.path.join(root, f"loop_{final_loop}.traj")
        final = read(final_path)

        # Try best.traj
        best_path = os.path.join(root, best_file)
        best = read(best_path) if os.path.exists(best_path) else None

        ncols = 3 if best is not None else 2
        fig, axes = plt.subplots(1, ncols, figsize=(5 * ncols, 5))

        plot_atoms(initial, axes[0], radii=0.5, show_unit_cell=2)
        axes[0].set_title(f"Initial\nE = {_safe_energy(initial):.6f} eV")

        plot_atoms(final, axes[1], radii=0.5, show_unit_cell=2)
        axes[1].set_title(f"Final (Loop {final_loop})\nE = {_safe_energy(final):.6f} eV")

        if best is not None:
            plot_atoms(best, axes[2], radii=0.5, show_unit_cell=2)
            axes[2].set_title(f"Best\nE = {_safe_energy(best):.6f} eV")
            
        
        Z = initial.get_atomic_numbers()
        unique_Z = sorted(set(Z))
        handles = []
        for z in unique_Z:
            c = jmol_colors[z]
            handles.append(mpatches.Patch(color=c, label=f'Z={z}'))

        axes[0].legend(handles=handles, loc='upper right', title='Element numbers', fontsize='small')        

        plt.tight_layout()

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            outpath = os.path.join(save_dir, f"compare_initial_final.{img_format}")
            plt.savefig(outpath, dpi=dpi, bbox_inches="tight")
            print(f"[Saved] {outpath}")
            if not show:
                plt.close(fig)

        if show:
            plt.show()

    except Exception as e:
        print(f"Error comparing structures: {e}")


def export_metrics(
    out_json: str = None,
    out_csv: str = None,
    loop_numbers: List[int] = None,
    energies: List[float] = None,
    volumes: List[float] = None,
    rmsds: List[float] = None,
):
    """Write metrics to JSON/CSV (if paths are provided)."""
    data = {
        "loop": loop_numbers or [],
        "energy_eV": energies or [],
        "volume_A3": volumes or [],
        "rmsd_A": rmsds or [],
    }

    if out_json:
        with open(out_json, "w") as f:
            json.dump(data, f, indent=2)
        print(f"[Saved] {out_json}")

    if out_csv:
        with open(out_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["loop", "energy_eV", "volume_A3", "rmsd_A"])
            # zip_longest-style safe write
            maxlen = max(len(data["loop"]), len(data["energy_eV"]), len(data["volume_A3"]), len(data["rmsd_A"]))
            for i in range(maxlen):
                row = [
                    data["loop"][i] if i < len(data["loop"]) else "",
                    data["energy_eV"][i] if i < len(data["energy_eV"]) else "",
                    data["volume_A3"][i] if i < len(data["volume_A3"]) else "",
                    data["rmsd_A"][i] if i < len(data["rmsd_A"]) else "",
                ]
                writer.writerow(row)
        print(f"[Saved] {out_csv}")


def diagnose_stuck_optimization(
    root: str = ".",
    pattern: str = "loop_*.traj",
    stuck_threshold: float = 1e-5,
    small_change_threshold: float = 0.1,
):
    """High-level diagnosis summary in the console."""
    traj_files = find_traj_files(root, pattern)
    if not traj_files:
        print("No trajectory files found for diagnosis.")
        return

    energies, volumes, structures = [], [], []
    loop_numbers = []

    for traj_file in traj_files:
        try:
            atoms = read(traj_file)
            energies.append(_safe_energy(atoms))
            volumes.append(atoms.get_volume())
            base = os.path.basename(traj_file)
            loop_numbers.append(int(base.split("_")[1].split(".")[0]))
        except Exception:
            pass

    if len(energies) < 2:
        print("Not enough data to diagnose")
        return

    print("\n" + "=" * 60)
    print("DIAGNOSIS")
    print("=" * 60)

    energy_changes = np.abs(np.diff(energies))
    stuck_count = int(np.sum(energy_changes < stuck_threshold))

    if stuck_count > len(energies) * 0.5:
        print("⚠️  ENERGY IS STUCK!")
        print(f"   {stuck_count}/{len(energies)-1} loops show minimal energy change")
        print("   Possible causes:")
        print("   - Temperature too low for annealing")
        print("   - Annealing time too short")
        print("   - System trapped in deep local minimum")
        print("   - Need more aggressive perturbations")

    volume_changes = np.abs(np.diff(volumes))
    if np.all(volume_changes < 1e-3):
        print("⚠️  CELL VOLUME NOT CHANGING!")
        print("   The cell might be too constrained")

    unique_energies = len(np.unique(np.round(energies, 6)))
    if unique_energies < len(energies) * 0.7:
        print("⚠️  REPEATED ENERGIES DETECTED!")
        print(f"   Only {unique_energies} unique energies out of {len(energies)} loops")
        print("   System is likely cycling between same configurations")

    print("\n📋 SUGGESTIONS:")
    if stuck_count > 0:
        print("   • Increase annealing temperature (T_peak)")
        print("   • Increase annealing time (T_hold_ps)")
        print("   • Add stronger random perturbations")
        print("   • Use simulated annealing with slower cooling")
    print("   • Try different starting configurations")
    print("   • Consider using basin hopping algorithm")
    print("   • Check if the force tolerance (fmax) is appropriate")


def main():
    p = argparse.ArgumentParser(description="Analyze MD/relax optimization outputs (loop_*.traj).")
    p.add_argument("--root", default=".", help="Directory containing traj files (default: .)")
    p.add_argument("--pattern", default="loop_*.traj", help='Filename pattern (default: "loop_*.traj")')
    p.add_argument("--stuck-threshold", type=float, default=1e-6, help="|ΔE| below this is considered stuck (default: 1e-6 eV)")
    p.add_argument("--small-change-threshold", type=float, default=0.1, help="RMSD below this is small (Å; default: 0.1)")
    p.add_argument("--save-dir", default=None, help="If set, save plots here")
    p.add_argument("--img-format", default="png", choices=["png", "svg", "pdf"], help="Image format (default: png)")
    p.add_argument("--dpi", type=int, default=150, help="Figure DPI (default: 150)")
    p.add_argument("--no-show", action="store_true", help="Do not display plots (useful for headless CI)")
    p.add_argument("--export-json", default=None, help="Write metrics JSON to this path")
    p.add_argument("--export-csv", default=None, help="Write metrics CSV to this path")
    p.add_argument("--compare", action="store_true", help="Also render initial vs final/best comparison")
    p.add_argument("--initial-file", default="loop_0.traj", help='Initial traj filename (default: "loop_0.traj")')
    p.add_argument("--best-file", default="best.traj", help='Best traj filename (default: "best.traj")')

    args = p.parse_args()
    show_plots = not args.no_show

    # Main analysis
    energies, volumes, structures = analyze_optimization_progress(
        root=args.root,
        pattern=args.pattern,
        stuck_threshold=args.stuck_threshold,
        small_change_threshold=args.small_change_threshold,
        save_dir=args.save_dir,
        show=show_plots,
        img_format=args.img_format,
        dpi=args.dpi,
    )

    # Export metrics if requested
    if energies:
        # Recompute loop numbers from files to export
        loop_files = find_traj_files(args.root, args.pattern)
        loop_numbers = [int(os.path.basename(f).split("_")[1].split(".")[0]) for f in loop_files]
        # Compute RMSDs again for export (consistent with last call)
        rmsds = []
        for i in range(1, len(structures)):
            p1 = structures[i - 1].get_positions()
            p2 = structures[i].get_positions()
            rmsds.append(float(np.sqrt(np.mean((p1 - p2) ** 2))))
        export_metrics(args.export_json, args.export_csv, loop_numbers, energies, volumes, rmsds)

    # Optional structural comparison
    if args.compare:
        compare_initial_vs_final(
            root=args.root,
            pattern=args.pattern,
            initial_file=args.initial_file,
            best_file=args.best_file,
            save_dir=args.save_dir,
            show=show_plots,
            img_format=args.img_format,
            dpi=args.dpi,
        )


if __name__ == "__main__":
    main()
