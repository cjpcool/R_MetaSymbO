"""
Enhanced script to load and visualize saved .traj files from structure optimization.
This script can visualize atomic structures with adaptive sizing for better clarity.
"""

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from ase.io import read, Trajectory
from ase.visualize.plot import plot_atoms
from ase.visualize import view
import warnings
warnings.filterwarnings("ignore")

def load_traj_file(filename):
    """Load a trajectory file and return the atoms object(s)."""
    try:
        # Try reading as trajectory first
        traj = read(filename, ':')  # ':' means read all frames
        return traj
    except:
        # If that fails, try reading as single structure
        atoms = read(filename)
        return [atoms]

def calculate_adaptive_radii(atoms, base_radii=None):
    """计算自适应的原子半径，基于原子间距离和体系大小"""
    from ase.data import covalent_radii
    from ase.neighborlist import NeighborList
    
    if base_radii is None:
        # 使用共价半径作为基础
        base_radii = [covalent_radii[atom.number] for atom in atoms]
    
    # 计算最近邻距离
    try:
        nl = NeighborList([1.5] * len(atoms), self_interaction=False, bothways=True)
        nl.update(atoms)
        
        min_distances = []
        for i in range(len(atoms)):
            indices, offsets = nl.get_neighbors(i)
            if len(indices) > 0:
                distances = atoms.get_distances(i, indices, mic=True)
                min_distances.append(np.min(distances))
        
        if min_distances:
            avg_min_distance = np.mean(min_distances)
            # 调整半径为最近邻距离的一定比例
            scale_factor = min(avg_min_distance / 4.0, 1.0)  # 不超过原始大小
            adaptive_radii = [r * scale_factor for r in base_radii]
        else:
            adaptive_radii = base_radii
            
    except:
        # 如果邻居列表计算失败，使用基于体系大小的简单缩放
        cell_volume = atoms.get_volume()
        n_atoms = len(atoms)
        
        # 估算原子密度并调整半径
        if n_atoms > 0 and cell_volume > 0:
            atom_density = n_atoms / cell_volume
            # 原子密度越高，半径应该越小
            scale_factor = min(1.0 / (atom_density ** (1/3) * 0.1), 2.0)
            adaptive_radii = [r * scale_factor for r in base_radii]
        else:
            adaptive_radii = base_radii
    
    # 确保半径在合理范围内
    adaptive_radii = [max(0.2, min(r, 2.0)) for r in adaptive_radii]
    
    return adaptive_radii

def visualize_single_structure(atoms, title="Atomic Structure", adaptive_size=True):
    """Visualize a single atomic structure with adaptive sizing."""
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # 计算自适应参数
    if adaptive_size:
        radii = calculate_adaptive_radii(atoms)
    else:
        radii = 0.5
    
    # 获取能量信息（安全处理）
    try:
        energy = atoms.get_potential_energy()
        energy_str = f"Energy: {energy:.6f} eV"
        
        # 如果能量异常高，添加警告
        if abs(energy) > 1000:
            energy_str += " ⚠️ HIGH ENERGY"
        elif abs(energy) > 100:
            energy_str += " ⚠️ ELEVATED ENERGY"
            
    except:
        energy_str = "Energy: N/A"
    
    # 绘制原子结构
    try:
        plot_atoms(atoms, ax, radii=radii, colors=None, show_unit_cell=2)
    except:
        # 如果自适应半径失败，使用固定半径
        plot_atoms(atoms, ax, radii=0.3, colors=None, show_unit_cell=2)
    
    # 设置标题和其他信息
    volume = atoms.get_volume()
    n_atoms = len(atoms)
    symbols = atoms.get_chemical_symbols()
    composition = {sym: symbols.count(sym) for sym in set(symbols)}
    comp_str = ', '.join([f"{sym}:{count}" for sym, count in composition.items()])
    
    full_title = f"{title}\n{energy_str}\nVolume: {volume:.2f} Ų, Atoms: {comp_str}"
    ax.set_title(full_title, fontsize=12)
    
    # 调整视图以更好地显示结构
    ax.set_aspect('equal')
    
    plt.tight_layout()
    return fig

def advanced_structure_visualization(atoms, title="Advanced Visualization"):
    """高级结构可视化，包含多种视图和分析"""
    fig = plt.figure(figsize=(16, 12))
    
    # 主结构视图 - 自适应大小
    ax1 = plt.subplot(2, 3, (1, 2))
    radii = calculate_adaptive_radii(atoms)
    
    try:
        plot_atoms(atoms, ax1, radii=radii, colors=None, show_unit_cell=2)
    except:
        plot_atoms(atoms, ax1, radii=0.3, colors=None, show_unit_cell=2)
    
    # 获取结构信息
    try:
        energy = atoms.get_potential_energy()
        energy_str = f"Energy: {energy:.6f} eV"
        if abs(energy) > 1000:
            energy_str += " ⚠️ VERY HIGH ENERGY"
        elif abs(energy) > 100:
            energy_str += " ⚠️ HIGH ENERGY"
    except:
        energy_str = "Energy: N/A"
    
    volume = atoms.get_volume()
    n_atoms = len(atoms)
    symbols = atoms.get_chemical_symbols()
    composition = {sym: symbols.count(sym) for sym in set(symbols)}
    
    ax1.set_title(f"{title}\n{energy_str}\nVolume: {volume:.2f} Ų", fontsize=12)
    ax1.set_aspect('equal')
    
    # 原子间距离分析
    ax2 = plt.subplot(2, 3, 3)
    try:
        from ase.neighborlist import neighbor_list
        i, j, d = neighbor_list('ijd', atoms, 3.0)  # 3Å cutoff
        
        if len(d) > 0:
            ax2.hist(d, bins=20, alpha=0.7, edgecolor='black')
            ax2.set_xlabel('Distance (Å)')
            ax2.set_ylabel('Count')
            ax2.set_title('Interatomic Distances')
            ax2.axvline(np.mean(d), color='red', linestyle='--', 
                       label=f'Mean: {np.mean(d):.2f} Å')
            ax2.legend()
            
            # 标记异常短的距离
            short_distances = d[d < 1.5]
            if len(short_distances) > 0:
                ax2.axvline(1.5, color='orange', linestyle=':', 
                           label=f'Short bonds: {len(short_distances)}')
                ax2.legend()
        else:
            ax2.text(0.5, 0.5, 'No neighbors\nwithin 3Å', 
                    ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title('Interatomic Distances')
    except:
        ax2.text(0.5, 0.5, 'Distance analysis\nfailed', 
                ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title('Interatomic Distances')
    
    # 原子坐标分布
    ax3 = plt.subplot(2, 3, 4)
    positions = atoms.get_positions()
    ax3.scatter(positions[:, 0], positions[:, 1], c=positions[:, 2], 
               s=50, alpha=0.7, cmap='viridis')
    ax3.set_xlabel('X (Å)')
    ax3.set_ylabel('Y (Å)')
    ax3.set_title('Atomic Positions (X-Y projection)')
    ax3.set_aspect('equal')
    
    # 体系信息表
    ax4 = plt.subplot(2, 3, 5)
    ax4.axis('off')
    
    info_text = f"""
SYSTEM INFORMATION
─────────────────────
Composition: {', '.join([f'{sym}:{count}' for sym, count in composition.items()])}
Total atoms: {n_atoms}
Cell volume: {volume:.2f} Ų
Density: {n_atoms/volume:.3f} atoms/Ų

STRUCTURE QUALITY
─────────────────────
"""
    
    # 检查结构质量
    try:
        forces = atoms.get_forces()
        max_force = np.max(np.linalg.norm(forces, axis=1))
        info_text += f"Max force: {max_force:.3f} eV/Å\n"
        
        if max_force > 5.0:
            info_text += "⚠️ Very high forces!\n"
        elif max_force > 1.0:
            info_text += "⚠️ High forces\n"
        else:
            info_text += "✓ Reasonable forces\n"
    except:
        info_text += "Forces: N/A\n"
    
    # 检查原子间距
    try:
        if len(d) > 0:
            min_dist = np.min(d)
            info_text += f"Min distance: {min_dist:.3f} Å\n"
            if min_dist < 1.0:
                info_text += "⚠️ Very short bonds!\n"
            elif min_dist < 1.5:
                info_text += "⚠️ Short bonds\n"
            else:
                info_text += "✓ Reasonable bonds\n"
    except:
        pass
    
    ax4.text(0.05, 0.95, info_text, transform=ax4.transAxes, 
            fontsize=10, verticalalignment='top', fontfamily='monospace')
    
    # 能量信息
    ax5 = plt.subplot(2, 3, 6)
    ax5.axis('off')
    
    try:
        energy_per_atom = energy / n_atoms
        energy_info = f"""
ENERGY ANALYSIS
─────────────────────
Total energy: {energy:.6f} eV
Energy/atom: {energy_per_atom:.6f} eV/atom

ENERGY ASSESSMENT
─────────────────────
"""
        
        if abs(energy_per_atom) < 5:
            energy_info += "✓ Normal energy range\n"
        elif abs(energy_per_atom) < 20:
            energy_info += "⚠️ Elevated energy\n"
        else:
            energy_info += "⚠️ Very high energy!\n"
            energy_info += "  • Check structure\n"
            energy_info += "  • Possible overlaps\n"
            energy_info += "  • Needs relaxation\n"
        
        ax5.text(0.05, 0.95, energy_info, transform=ax5.transAxes,
                fontsize=10, verticalalignment='top', fontfamily='monospace')
    except:
        ax5.text(0.05, 0.95, "Energy analysis\nnot available", 
                transform=ax5.transAxes, fontsize=10, verticalalignment='top')
    
    plt.tight_layout()
    return fig

def plot_energy_evolution(traj_files):
    """Plot energy evolution across different trajectory files."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    all_energies = []
    labels = []
    
    for traj_file in sorted(traj_files):
        try:
            atoms_list = load_traj_file(traj_file)
            energies = [atoms.get_potential_energy() for atoms in atoms_list]
            all_energies.extend(energies)
            
            # Plot this trajectory
            x_start = len(all_energies) - len(energies)
            x_end = len(all_energies)
            ax.plot(range(x_start, x_end), energies, 'o-', 
                   label=f"{os.path.basename(traj_file)}")
            
        except Exception as e:
            print(f"Error reading {traj_file}: {e}")
    
    ax.set_xlabel('Step')
    ax.set_ylabel('Potential Energy (eV)')
    ax.set_title('Energy Evolution During Optimization')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig

def compare_structures(traj_files, max_structures=5, adaptive_size=True):
    """Compare multiple structures side by side with adaptive sizing."""
    n_files = min(len(traj_files), max_structures)
    fig, axes = plt.subplots(1, n_files, figsize=(6*n_files, 6))
    
    if n_files == 1:
        axes = [axes]
    
    for i, traj_file in enumerate(sorted(traj_files)[:max_structures]):
        try:
            atoms_list = load_traj_file(traj_file)
            atoms = atoms_list[-1]  # Take the last (final) structure
            
            # 计算自适应半径
            if adaptive_size:
                radii = calculate_adaptive_radii(atoms)
            else:
                radii = 0.5
            
            try:
                plot_atoms(atoms, axes[i], radii=radii, colors=None, show_unit_cell=2)
            except:
                plot_atoms(atoms, axes[i], radii=0.3, colors=None, show_unit_cell=2)
            
            # 获取能量和其他信息
            try:
                energy = atoms.get_potential_energy()
                energy_str = f"E: {energy:.4f} eV"
                if abs(energy) > 100:
                    energy_str += " ⚠️"
            except:
                energy_str = "E: N/A"
            
            volume = atoms.get_volume()
            axes[i].set_title(f"{os.path.basename(traj_file)}\n{energy_str}\nV: {volume:.1f} Ų", 
                            fontsize=10)
            axes[i].set_aspect('equal')
            
        except Exception as e:
            print(f"Error reading {traj_file}: {e}")
            axes[i].set_title(f"Error: {os.path.basename(traj_file)}")
    
    plt.tight_layout()
    return fig

def animate_trajectory(traj_file, save_gif=False, adaptive_size=True):
    """Create an animation of a trajectory (if multiple frames exist) with adaptive sizing."""
    try:
        atoms_list = load_traj_file(traj_file)
        
        if len(atoms_list) == 1:
            print(f"Only one frame in {traj_file}, showing static structure")
            return visualize_single_structure(atoms_list[0], 
                                           f"Structure from {os.path.basename(traj_file)}", 
                                           adaptive_size=adaptive_size)
        
        # Show a few key frames
        n_frames = len(atoms_list)
        key_frames = [0, n_frames//4, n_frames//2, 3*n_frames//4, n_frames-1]
        
        fig, axes = plt.subplots(1, len(key_frames), figsize=(5*len(key_frames), 5))
        if len(key_frames) == 1:
            axes = [axes]
        
        # 计算所有帧的统一缩放因子（基于第一帧和最后一帧）
        if adaptive_size:
            first_radii = calculate_adaptive_radii(atoms_list[0])
            last_radii = calculate_adaptive_radii(atoms_list[-1])
            # 使用平均半径确保一致性
            avg_radius = (np.mean(first_radii) + np.mean(last_radii)) / 2
            unified_radii = avg_radius
        else:
            unified_radii = 0.5
        
        for i, frame_idx in enumerate(key_frames):
            atoms = atoms_list[frame_idx]
            
            try:
                plot_atoms(atoms, axes[i], radii=unified_radii, colors=None, show_unit_cell=2)
            except:
                plot_atoms(atoms, axes[i], radii=0.3, colors=None, show_unit_cell=2)
            
            try:
                energy = atoms.get_potential_energy()
                energy_str = f"E: {energy:.4f} eV"
                if abs(energy) > 100:
                    energy_str += " ⚠️"
            except:
                energy_str = "E: N/A"
                
            axes[i].set_title(f"Frame {frame_idx+1}/{n_frames}\n{energy_str}", fontsize=10)
            axes[i].set_aspect('equal')
        
        plt.suptitle(f"Key frames from {os.path.basename(traj_file)}", fontsize=14)
        plt.tight_layout()
        return fig
        
    except Exception as e:
        print(f"Error animating {traj_file}: {e}")
        return None

def analyze_final_structures(traj_files):
    """Analyze and compare final structures from all trajectory files."""
    results = []
    
    for traj_file in traj_files:
        try:
            atoms_list = load_traj_file(traj_file)
            final_atoms = atoms_list[-1]
            
            info = {
                'file': os.path.basename(traj_file),
                'energy': final_atoms.get_potential_energy(),
                'n_atoms': len(final_atoms),
                'cell_volume': final_atoms.get_volume(),
                'symbols': final_atoms.get_chemical_symbols()
            }
            results.append(info)
            
        except Exception as e:
            print(f"Error analyzing {traj_file}: {e}")
    
    # Sort by energy
    results.sort(key=lambda x: x['energy'])
    
    print("\n" + "="*60)
    print("FINAL STRUCTURE ANALYSIS")
    print("="*60)
    print(f"{'File':<20} {'Energy (eV)':<15} {'N_atoms':<10} {'Volume':<15}")
    print("-"*60)
    
    for result in results:
        print(f"{result['file']:<20} {result['energy']:<15.6f} {result['n_atoms']:<10} {result['cell_volume']:<15.3f}")
    
    print(f"\nBest structure: {results[0]['file']} with energy {results[0]['energy']:.6f} eV")
    
    return results

def main():
    """Main function to visualize trajectory files."""
    print("Looking for .traj files in current directory...")
    
    # Find all .traj files
    traj_files = glob.glob("*.traj")
    
    if not traj_files:
        print("No .traj files found in current directory!")
        return
    
    print(f"Found {len(traj_files)} trajectory files: {traj_files}")
    
    # Analyze final structures
    results = analyze_final_structures(traj_files)
    
    # Plot energy evolution
    if len(traj_files) > 1:
        print("\nPlotting energy evolution...")
        fig_energy = plot_energy_evolution(traj_files)
        plt.show()
        
        # Compare structures
        print("Comparing final structures...")
        fig_compare = compare_structures(traj_files, adaptive_size=True)
        plt.show()
    
    # Visualize best structure with advanced analysis
    if results:
        best_file = results[0]['file']
        print(f"\nVisualizing best structure from {best_file}...")
        best_atoms = load_traj_file(best_file)[-1]
        
        # Standard visualization
        fig_best = visualize_single_structure(best_atoms, f"Best Structure ({best_file})", adaptive_size=True)
        plt.show()
        
        # Advanced visualization
        print("Showing advanced analysis...")
        fig_advanced = advanced_structure_visualization(best_atoms, f"Advanced Analysis ({best_file})")
        plt.show()
        
        # Try to open in ASE viewer (if available)
        try:
            print("Opening best structure in ASE viewer...")
            view(best_atoms)
        except:
            print("ASE viewer not available or failed to open")
    
    # Animate trajectories (show key frames)
    print("\nShowing trajectory evolution...")
    for traj_file in traj_files[:]:  # Limit to first 3 files
        fig_anim = animate_trajectory(traj_file, adaptive_size=True)
        if fig_anim:
            plt.show()

if __name__ == "__main__":
    main()
