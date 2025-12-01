"""
测试自适应可视化效果的脚本
"""

from ase.build import bulk
from ase.calculators.emt import EMT
import numpy as np
import matplotlib.pyplot as plt
from visualize_traj_enhanced import visualize_single_structure, advanced_structure_visualization

def create_test_structures():
    """创建测试结构，包括正常和异常情况"""
    
    # 1. 正常的Al-Cu合金结构
    normal_atoms = bulk('Al', 'fcc', a=4.05).repeat((2,2,2))
    normal_atoms[0].symbol = 'Cu'
    normal_atoms[3].symbol = 'Cu'
    normal_atoms.calc = EMT()
    
    # 2. 创建一个压缩的结构（高能量）
    compressed_atoms = normal_atoms.copy()
    cell = compressed_atoms.get_cell()
    compressed_atoms.set_cell(cell * 0.8, scale_atoms=True)  # 压缩20%
    compressed_atoms.calc = EMT()
    
    # 3. 创建一个拉伸的结构
    stretched_atoms = normal_atoms.copy()
    cell = stretched_atoms.get_cell()
    stretched_atoms.set_cell(cell * 1.3, scale_atoms=True)  # 拉伸30%
    stretched_atoms.calc = EMT()
    
    # 4. 创建一个有原子重叠的结构（极高能量）
    overlapped_atoms = normal_atoms.copy()
    positions = overlapped_atoms.get_positions()
    # 移动一个原子使其与另一个重叠
    positions[1] = positions[0] + 0.1  # 非常接近重叠
    overlapped_atoms.set_positions(positions)
    overlapped_atoms.calc = EMT()
    
    return {
        'normal': normal_atoms,
        'compressed': compressed_atoms, 
        'stretched': stretched_atoms,
        'overlapped': overlapped_atoms
    }

def compare_visualization_methods():
    """比较固定半径和自适应半径的可视化效果"""
    
    structures = create_test_structures()
    
    print("创建测试结构...")
    for name, atoms in structures.items():
        energy = atoms.get_potential_energy()
        volume = atoms.get_volume()
        print(f"{name:12}: E = {energy:8.3f} eV, V = {volume:6.1f} Ų")
    
    # 比较正常结构和压缩结构
    print("\n比较可视化方法...")
    
    for structure_name in ['normal', 'compressed', 'overlapped']:
        atoms = structures[structure_name]
        
        print(f"\n处理 {structure_name} 结构...")
        
        # 固定半径可视化
        fig1 = visualize_single_structure(atoms, 
                                        f"{structure_name.title()} Structure (Fixed Radii)", 
                                        adaptive_size=False)
        plt.show()
        
        # 自适应半径可视化
        fig2 = visualize_single_structure(atoms, 
                                        f"{structure_name.title()} Structure (Adaptive Radii)", 
                                        adaptive_size=True)
        plt.show()
        
        # 高级分析（仅对有问题的结构）
        if structure_name in ['compressed', 'overlapped']:
            print(f"显示 {structure_name} 结构的高级分析...")
            fig3 = advanced_structure_visualization(atoms, 
                                                   f"{structure_name.title()} Advanced Analysis")
            plt.show()

def demonstrate_adaptive_features():
    """演示自适应功能的特点"""
    
    print("\n" + "="*60)
    print("自适应可视化功能演示")
    print("="*60)
    print("自适应可视化的优点:")
    print("1. 根据原子间距离自动调整原子大小")
    print("2. 防止高能量体系中原子显示过小")
    print("3. 避免原子重叠时的视觉混乱")
    print("4. 提供结构质量的直观反馈")
    print("5. 自动标识异常高能量的结构")
    print("\n高级分析功能:")
    print("1. 原子间距离分布分析")
    print("2. 结构质量评估")
    print("3. 能量异常检测")
    print("4. 力的大小分析")
    print("5. 原子坐标分布可视化")

if __name__ == "__main__":
    print("自适应结构可视化测试")
    print("="*50)
    
    demonstrate_adaptive_features()
    compare_visualization_methods()
    
    print("\n测试完成!")
    print("建议:")
    print("- 对于正常结构，固定和自适应半径效果相似")
    print("- 对于异常结构，自适应半径提供更好的可视化")
    print("- 高级分析能够识别结构问题并提供诊断信息")
