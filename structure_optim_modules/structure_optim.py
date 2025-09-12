from ase import units
from ase.io import read, write
from ase.optimize import FIRE, BFGS
from ase.constraints import UnitCellFilter, ExpCellFilter
from ase.md.langevin import Langevin
from ase.md.nptberendsen import NPTBerendsen
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary, ZeroRotation


from ase.io import write


import warnings
warnings.filterwarnings("ignore")





B_GPa = 82.0
kappa = 1.0 / (B_GPa * units.GPa)             # ASE 需要 1/(eV/Å^3)


def relax_0K(atoms, fmax=0.03, relax_cell=False, logfile='relax.log'):
    """0 K relaxation to nearest local minimum."""
    if relax_cell:
        ecf = ExpCellFilter(atoms, hydrostatic_strain=True)  # or UnitCellFilter
        opt = BFGS(ecf, logfile=logfile)
    else:
        opt = FIRE(atoms, logfile=logfile)
    opt.run(fmax=fmax)
    return atoms

def md_anneal(atoms, T_start=300, T_peak=900, T_hold_ps=10, npt=False,
              pressure_atm=1.0, tstep_fs=2.0, friction=0.01):
    """Heat to T_peak, hold, optionally at constant pressure."""
    # Initialize momenta & remove overall drift/rotation
    MaxwellBoltzmannDistribution(atoms, temperature_K=T_start)
    Stationary(atoms); ZeroRotation(atoms)

    dt = tstep_fs * units.fs
    if npt:
        md = NPTBerendsen(atoms, dt,
                          temperature_K=T_peak, taut=100*units.fs,
                          pressure=pressure_atm * 1.01325e5*units.Pascal, taup=1000*units.fs,
                          compressibility=kappa)
    else:
        md = Langevin(atoms, dt, temperature_K=T_peak, friction=friction)

    steps_hold = int(T_hold_ps*1000/tstep_fs)
    md.run(steps_hold)
    return atoms

def md_quench(atoms, T_final=100, cool_ps=20, tstep_fs=2.0, npt=False,
              pressure_atm=1.0, friction=0.01):
    """Cool from current T to T_final; then zero momenta."""
    dt = tstep_fs * units.fs
    if npt:
        md = NPTBerendsen(atoms, dt,
                          temperature_K=T_final, taut=100*units.fs,
                          pressure=pressure_atm * 1.01325e5*units.Pascal, taup=1000*units.fs,
                          compressibility=kappa)
    else:
        md = Langevin(atoms, dt, temperature_K=T_final, friction=friction)

    steps_cool = int(cool_ps*1000/tstep_fs)
    md.run(steps_cool)

    # Quench to 0 K and remove kinetic energy
    atoms.set_momenta(0.0 * atoms.get_momenta())
    return atoms

def optimize_with_md_loop(atoms, calc, n_loops=5, fmax=0.03, relax_cell=False,
                          anneal_kwargs=None, quench_kwargs=None,
                          min_deltaE=1e-3):
    """Global-ish search: anneal → quench → relax, keep best structure."""
    import numpy as np
    
    atoms.calc = calc
    best = atoms.copy()
    best_E = atoms.get_potential_energy()
    print("Initial potential energy:", best_E)
    # Pre-relax to get a sane starting point
    write(f'./loop_0.traj', atoms)

    print("Pre-relaxing to 0 K...")
    relax_0K(atoms, fmax=fmax, relax_cell=relax_cell)
    print("Initial potential energy after pre-relax:", atoms.get_potential_energy())

    # Update best if pre-relaxation improved the structure
    E_prerelax = atoms.get_potential_energy()
    if E_prerelax < best_E:
        best_E = E_prerelax
        best = atoms.copy()

    write(f'./loop_1.traj', atoms)

    print("Starting optimization loop...")
    for i in range(n_loops):
        print(f"Loop {i+2}/{n_loops}: Annealing and quenching...")
        
        # Start each loop from the best structure found so far
        atoms = best.copy()
        atoms.calc = calc
        
        # Add small random perturbation to escape local minima
        positions = atoms.get_positions()
        noise_amplitude = 0.1  # Angstrom
        noise = np.random.normal(0, noise_amplitude, positions.shape)
        atoms.set_positions(positions + noise)
        
        # If relax_cell is True, also add small cell perturbation
        if relax_cell:
            cell = atoms.get_cell()
            cell_noise = np.random.normal(0, 0.02, cell.shape)  # 2% strain noise
            atoms.set_cell(cell * (1 + cell_noise), scale_atoms=True)
        
        print(f"  Starting energy after perturbation: {atoms.get_potential_energy():.6f} eV")
        
        md_anneal(atoms, **(anneal_kwargs or {}))
        md_quench(atoms, **(quench_kwargs or {}))
        print("Relaxing to 0 K...")
        relax_0K(atoms, fmax=fmax, relax_cell=relax_cell)

        E = atoms.get_potential_energy()
        print(f"Loop {i+1}/{n_loops}: Final energy = {E:.6f} eV (best so far: {best_E:.6f} eV)")
        write(f'./loop_{i+1}.traj', atoms)
        
        if E < best_E - min_deltaE:
            improvement = best_E - E
            best_E = E
            best = atoms.copy()
            print(f"  *** NEW BEST STRUCTURE! Improvement: {improvement:.6f} eV ***")
        else:
            print(f"  No improvement (ΔE = {E - best_E:.6f} eV)")

    print(f"\nOptimization completed. Best energy: {best_E:.6f} eV")
    return best, best_E




if __name__ == "__main__":
    from ase.build import bulk, molecule
    from ase.spacegroup import crystal
    from ase.calculators.emt import EMT

    from ase.visualize.plot import plot_atoms
    import matplotlib.pyplot as plt
    from ase.visualize import view
    
    atoms = bulk('Al', 'fcc', a=4.05).repeat((2,2,2))

    print(atoms)
    # 替换部分 Al 为 Cu
    atoms[0].symbol = 'Cu'
    atoms[3].symbol = 'Cu'
    atoms[7].symbol = 'Cu'

    # atoms = molecule('H2O')
    calc  = EMT()                           # swap to EAM/Tersoff/ReaxFF/DFT/etc.
    view(atoms) 
    

    best_atoms, best_E = optimize_with_md_loop(
    atoms, calc,
    n_loops=6,
    fmax=0.03,
    relax_cell=True,
    anneal_kwargs=dict(T_start=300, T_peak=1200, T_hold_ps=20, npt=False, pressure_atm=1.0, tstep_fs=1.0),
    quench_kwargs=dict(T_final=100, cool_ps=40, npt=False, pressure_atm=1.0, tstep_fs=1.0),
    min_deltaE=1e-4
    )   

    from ase.io import write
    write('./best.traj', best_atoms)
