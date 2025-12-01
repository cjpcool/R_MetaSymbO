#!/usr/bin/env python3
"""
Batch Runs Aggregator & Visualizer
==================================

Scans ./batch_runs/claim_XXXX/ for:
  - timings.json
  - mdopt/summary.json

Produces:
  1) Statistical analysis of time in each step (gen, uma, dft, analysis)
  2) UMA vs DFT comparisons (energies and any overlapping numeric properties)
  3) Distributions for each numeric property present in summary.json (by method)

Outputs saved under --outdir (default: ./batch_runs/analysis_summary):
  - times.csv, times_stats.csv
  - energies.csv
  - properties_long.csv
  - comparisons.csv (UMA vs DFT property pairs + delta)
  - plots/*.png (histograms, boxplots, scatter, per-property plots)
  - index.json (quick summary)

Dependencies: pandas, numpy, matplotlib, seaborn

Usage:
  python analyze_batch_runs.py \
      --batch-root ./batch_runs \
      --outdir ./batch_runs/analysis_summary

Optionally add --show to display figures interactively after saving.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def _optional_imports():
    try:
        import pandas as pd  # type: ignore
    except Exception as e:
        raise RuntimeError("pandas is required: pip install pandas") from e
    try:
        import matplotlib.pyplot as plt  # type: ignore
        import seaborn as sns  # type: ignore
    except Exception as e:
        raise RuntimeError("matplotlib and seaborn are required: pip install matplotlib seaborn") from e
    return pd, plt, sns


def find_claim_dirs(batch_root: Path) -> List[Path]:
    return sorted([p for p in batch_root.glob('claim_*') if p.is_dir()])


def read_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        if path.is_file():
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
    except Exception:
        return None
    return None


def flatten_numeric(d: Dict[str, Any], prefix: str = "") -> Dict[str, float]:
    """Return a flat dict of numeric scalar values from a possibly nested dict.
    Lists/tuples are ignored unless they are 0-d numeric. Non-numerics skipped.
    Keys are joined by dots with optional prefix.
    """
    out: Dict[str, float] = {}
    def rec(obj: Any, key_prefix: str):
        if obj is None:
            return
        if isinstance(obj, (int, float)) and not isinstance(obj, bool):
            out[key_prefix.rstrip('.')] = float(obj)
            return
        if isinstance(obj, dict):
            for k, v in obj.items():
                rec(v, f"{key_prefix}{k}.")
            return
        # ignore sequences and others
    rec(d, prefix)
    return out


def aggregate(batch_root: Path) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Return tuples of rows for times, energies (from timings), and properties (from summary)."""
    claim_dirs = find_claim_dirs(batch_root)
    times_rows: List[Dict[str, Any]] = []
    energies_rows: List[Dict[str, Any]] = []
    props_rows: List[Dict[str, Any]] = []

    for cdir in claim_dirs:
        timings_path = cdir / 'timings.json'
        summary_path = cdir / 'mdopt' / 'summary.json'
        tdata = read_json(timings_path)
        sdata = read_json(summary_path)

        idx = None
        claim_text = None

        if tdata:
            idx = tdata.get('index')
            claim_text = tdata.get('claim')
            tsec = (tdata.get('timing_seconds') or {})
            times_rows.append({
                'claim_dir': str(cdir),
                'index': idx,
                'claim': claim_text,
                'gen_logged': tsec.get('gen_logged'),
                'gen_wall': tsec.get('gen_wall'),
                'uma_wall': tsec.get('uma_wall'),
                'dft_logged': tsec.get('dft_logged'),
                'analysis_wall': tsec.get('analysis_wall'),
            })

            energies = (tdata.get('energies') or {})
            energies_rows.append({
                'claim_dir': str(cdir),
                'index': idx,
                'claim': claim_text,
                'uma_energy_eV': energies.get('uma_energy_eV'),
                'dft_energy_eV': energies.get('dft_energy_eV'),
                'dft_gap_eV': energies.get('dft_gap_eV'),
            })

        if sdata:
            # Expect optional keys 'uma' and 'dft' but be robust
            for method in ['uma', 'dft']:
                md = sdata.get(method)
                if isinstance(md, dict):
                    flat = flatten_numeric(md)
                    for prop, val in flat.items():
                        props_rows.append({
                            'claim_dir': str(cdir),
                            'index': idx,
                            'claim': claim_text,
                            'method': method,
                            'property': prop,
                            'value': val,
                        })
            # Also consider top-level numeric fields if any
            top_flat = flatten_numeric({k: v for k, v in sdata.items() if k not in ('uma', 'dft')})
            for prop, val in top_flat.items():
                props_rows.append({
                    'claim_dir': str(cdir),
                    'index': idx,
                    'claim': claim_text,
                    'method': 'top',
                    'property': prop,
                    'value': val,
                })

    return times_rows, energies_rows, props_rows


def compute_comparisons(props_rows: List[Dict[str, Any]]):
    pd, _, _ = _optional_imports()
    if not props_rows:
        return pd.DataFrame(), pd.DataFrame()
    df = pd.DataFrame(props_rows)
    # focus on UMA/DFT
    dfud = df[df['method'].isin(['uma', 'dft'])].copy()
    if dfud.empty:
        return pd.DataFrame(), pd.DataFrame()
    wide = dfud.pivot_table(index=['claim_dir','index','claim','property'], columns='method', values='value', aggfunc='mean').reset_index()
    # columns may be MultiIndex depending on pandas version; ensure flat
    if isinstance(wide.columns, pd.MultiIndex):
        wide.columns = ['_'.join([str(c) for c in col if c != '']).strip('_') for col in wide.columns]
    # compute delta where both exist
    if 'uma' in wide.columns and 'dft' in wide.columns:
        wide['delta'] = wide['dft'] - wide['uma']
    else:
        # older column naming fallback
        if 'value_dft' in wide.columns and 'value_uma' in wide.columns:
            wide['delta'] = wide['value_dft'] - wide['value_uma']
            wide = wide.rename(columns={'value_dft':'dft','value_uma':'uma'})
        else:
            wide['delta'] = np.nan
    # Separate energy-only convenience table
    energy_cmp = wide[wide['property'].str.contains('energy', case=False, na=False)].copy()
    return wide, energy_cmp


def save_csv(df, path: Path):
    if df is not None and not getattr(df, 'empty', False):
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)


def compute_and_save_stats(times_df, outdir: Path):
    pd, _, _ = _optional_imports()
    stats = {}
    if times_df is None or times_df.empty:
        return stats
    numeric_cols = ['gen_logged','gen_wall','uma_wall','dft_logged','analysis_wall']
    desc = times_df[numeric_cols].describe().T
    save_csv(desc.reset_index().rename(columns={'index':'metric'}), outdir / 'times_stats.csv')
    stats['times'] = desc.to_dict()
    # totals
    times_df['total_wall_est'] = times_df[['gen_wall','uma_wall','analysis_wall']].sum(axis=1, skipna=True)
    desc_total = times_df['total_wall_est'].describe()
    save_csv(desc_total.reset_index().rename(columns={'index':'stat','total_wall_est':'value'}), outdir / 'total_wall_stats.csv')
    stats['total_wall_est'] = desc_total.to_dict()
    # save enriched times
    save_csv(times_df, outdir / 'times.csv')
    return stats


def plot_all(times_df, energies_df, props_df, cmp_df, energy_cmp_df, outdir: Path, show: bool = False, save_format: str = 'png'):
    pd, plt, sns = _optional_imports()
    sns.set(style='whitegrid')
    plots_dir = outdir / 'plots'
    plots_dir.mkdir(parents=True, exist_ok=True)

    # 1) Timing distributions
    if times_df is not None and not times_df.empty:
        time_cols = ['gen_logged','gen_wall','uma_wall','dft_logged','analysis_wall']
        available_cols = [c for c in time_cols if c in times_df.columns]
        if available_cols:
            # Histograms
            n = len(available_cols)
            fig, axes = plt.subplots(1, n, figsize=(5*n, 4), squeeze=False)
            for i, c in enumerate(available_cols):
                sns.histplot(times_df[c].dropna(), bins=20, kde=True, ax=axes[0, i])
                axes[0, i].set_title(f"{c} (s)")
            fig.tight_layout()
            fig.savefig(plots_dir / f'times_hist.{save_format}', dpi=200)
            if show: plt.show()
            plt.close(fig)

            # Boxplots
            fig, ax = plt.subplots(figsize=(max(6, 1.8*n), 4))
            sns.boxplot(data=times_df[available_cols], orient='h', ax=ax)
            ax.set_title('Timing distributions')
            fig.tight_layout()
            fig.savefig(plots_dir / f'times_box.{save_format}', dpi=200)
            if show: plt.show()
            plt.close(fig)

    # 2) UMA vs DFT comparisons (energies + general)
    if cmp_df is not None and not cmp_df.empty:
        # Generic: for each property with both UMA and DFT, plot scatter and delta hist
        props = sorted(cmp_df['property'].dropna().unique())
        for prop in props:
            sub = cmp_df[cmp_df['property'] == prop].dropna(subset=['uma','dft'])
            if sub.empty:
                continue
            # Scatter UMA vs DFT
            fig, ax = plt.subplots(figsize=(5.5, 5))
            sns.scatterplot(data=sub, x='uma', y='dft', ax=ax)
            lims = [np.nanmin([sub['uma'].min(), sub['dft'].min()]), np.nanmax([sub['uma'].max(), sub['dft'].max()])]
            if not (math.isinf(lims[0]) or math.isinf(lims[1])):
                pad = 0.05 * (lims[1] - lims[0] + 1e-12)
                ax.plot([lims[0]-pad, lims[1]+pad], [lims[0]-pad, lims[1]+pad], ls='--', c='gray', alpha=0.7)
                ax.set_xlim(lims[0]-pad, lims[1]+pad)
                ax.set_ylim(lims[0]-pad, lims[1]+pad)
            ax.set_title(f"UMA vs DFT: {prop}")
            ax.set_xlabel('UMA')
            ax.set_ylabel('DFT')
            fig.tight_layout()
            fig.savefig(plots_dir / f'cmp_scatter_{prop.replace("/","_")}.{save_format}', dpi=200)
            if show: plt.show()
            plt.close(fig)

            # Delta histogram (DFT - UMA)
            if 'delta' in sub.columns:
                fig, ax = plt.subplots(figsize=(5.5, 4))
                sns.histplot(sub['delta'].dropna(), bins=20, kde=True, ax=ax)
                ax.axvline(0.0, ls='--', c='gray')
                ax.set_title(f"Delta (DFT-UMA): {prop}")
                fig.tight_layout()
                fig.savefig(plots_dir / f'cmp_delta_hist_{prop.replace("/","_")}.{save_format}', dpi=200)
                if show: plt.show()
                plt.close(fig)

    # 3) Property distributions by method
    if props_df is not None and not props_df.empty:
        props = sorted(props_df['property'].dropna().unique())
        # If too many, still loop but save individually
        for prop in props:
            sub = props_df[props_df['property'] == prop]
            if sub.empty:
                continue
            # Hist by method
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.histplot(data=sub, x='value', hue='method', bins=20, kde=True, element='step', stat='density', common_norm=False, ax=ax)
            ax.set_title(f"Distribution: {prop}")
            fig.tight_layout()
            fig.savefig(plots_dir / f'prop_hist_{prop.replace("/","_")}.{save_format}', dpi=200)
            if show: plt.show()
            plt.close(fig)

            # Box by method
            fig, ax = plt.subplots(figsize=(5.5, 4))
            sns.boxplot(data=sub, x='method', y='value', ax=ax)
            ax.set_title(f"Box: {prop} by method")
            fig.tight_layout()
            fig.savefig(plots_dir / f'prop_box_{prop.replace("/","_")}.{save_format}', dpi=200)
            if show: plt.show()
            plt.close(fig)

    # 4) Simple energy scatter from energies_df if available
    if energies_df is not None and not energies_df.empty:
        if 'uma_energy_eV' in energies_df.columns and 'dft_energy_eV' in energies_df.columns:
            sub = energies_df.dropna(subset=['uma_energy_eV','dft_energy_eV'])
            if not sub.empty:
                fig, ax = plt.subplots(figsize=(5.5,5))
                sns.scatterplot(data=sub, x='uma_energy_eV', y='dft_energy_eV', ax=ax)
                lims = [np.nanmin([sub['uma_energy_eV'].min(), sub['dft_energy_eV'].min()]), np.nanmax([sub['uma_energy_eV'].max(), sub['dft_energy_eV'].max()])]
                pad = 0.05 * (lims[1] - lims[0] + 1e-12)
                ax.plot([lims[0]-pad, lims[1]+pad], [lims[0]-pad, lims[1]+pad], ls='--', c='gray', alpha=0.7)
                ax.set_title('UMA vs DFT energy (eV)')
                ax.set_xlabel('UMA energy (eV)')
                ax.set_ylabel('DFT energy (eV)')
                fig.tight_layout()
                fig.savefig(plots_dir / f'energy_scatter.{save_format}', dpi=200)
                if show: plt.show()
                plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description='Aggregate timings and summary.json across batch runs and visualize.')
    parser.add_argument('--batch-root', default='./batch_runs', help='Root directory containing claim_XXXX subfolders')
    parser.add_argument('--outdir', default='./batch_runs/analysis_summary', help='Directory to save outputs')
    parser.add_argument('--show', action='store_true', help='Show plots interactively after saving')
    parser.add_argument('--save-format', default='png', choices=['png','pdf','svg'], help='Image format for plots')
    args = parser.parse_args()

    batch_root = Path(args.batch_root).resolve()
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    # Aggregate
    times_rows, energies_rows, props_rows = aggregate(batch_root)
    pd, _, _ = _optional_imports()

    times_df = pd.DataFrame(times_rows) if times_rows else pd.DataFrame()
    energies_df = pd.DataFrame(energies_rows) if energies_rows else pd.DataFrame()
    props_df = pd.DataFrame(props_rows) if props_rows else pd.DataFrame()

    # Save raw CSVs
    save_csv(times_df, outdir / 'times.csv')
    save_csv(energies_df, outdir / 'energies.csv')
    save_csv(props_df, outdir / 'properties_long.csv')

    # Compute comparisons and stats
    cmp_df, energy_cmp_df = compute_comparisons(props_rows)
    save_csv(cmp_df, outdir / 'comparisons.csv')
    save_csv(energy_cmp_df, outdir / 'energy_comparisons.csv')
    compute_and_save_stats(times_df, outdir)

    # Plot
    plot_all(times_df, energies_df, props_df, cmp_df, energy_cmp_df, outdir, show=args.show, save_format=args.save_format)

    # Write index.json summary
    summary = {
        'counts': {
            'claims_found': len(find_claim_dirs(batch_root)),
            'timings_rows': len(times_rows),
            'energies_rows': len(energies_rows),
            'properties_rows': len(props_rows),
            'comparisons_rows': 0 if getattr(cmp_df, 'empty', True) else len(cmp_df),
        },
        'paths': {
            'outdir': str(outdir),
            'plots_dir': str(outdir / 'plots')
        },
        'notes': 'All plots saved under plots/. comparisons.csv contains per-property UMA vs DFT with delta.'
    }
    with open(outdir / 'index.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)

    print(f"Saved outputs under: {outdir}")


if __name__ == '__main__':
    main()
