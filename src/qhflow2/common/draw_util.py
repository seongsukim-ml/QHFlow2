import matplotlib.pyplot as plt

def matshow(matrix, max_val=None, max_mul=0.1, colorbar=True, frame=True, ticks=False, save_name=None, drawline=[], ratio=0.8):
    fig, ax = plt.subplots(figsize=(6*ratio,5*ratio))
    if matrix.ndim == 3:
        matrix = matrix[0]
    if max_val is None:
        max_val = abs(matrix).max() * max_mul
    im = ax.imshow(matrix, cmap='bwr',vmin=-max_val, vmax=max_val)
    if colorbar:
        fig.colorbar(im)
    if not ticks:
        ax.set_xticks([])
        ax.set_yticks([])
    
    spines_name = ['top', 'right', 'bottom', 'left']
    for spine in spines_name:
        ax.spines[spine].set_color('black')
        ax.spines[spine].set_linewidth(0.5)
    
    if not frame: 
        for spine in spines_name:
            ax.spines[spine].set_visible(False)
    
    cur = 0
    for idx, orb_num in enumerate(drawline):
        color = 'gray'
        ls = '--'
        lw = 0.3
        if orb_num < 0:
            orb_num = -orb_num
            color = 'black'
            ls = '-'
            lw = 0.5
        cur = cur + orb_num
        ax.axhline(y=cur-0.5, color=color, linestyle=ls, linewidth=lw)
        ax.axvline(x=cur-0.5, color=color, linestyle=ls, linewidth=lw)
    if save_name is not None:
        plt.savefig(save_name, bbox_inches='tight', dpi=600)
    plt.show()
    
    return max_val

# Visualize with py3Dmol
import py3Dmol

def atom_data_to_xyz(atomic_numbers, positions):
    """Convert atomic data to XYZ format string"""
    # Element symbols mapping
    element_map = {
        1: 'H', 2: 'He', 3: 'Li', 4: 'Be', 5: 'B', 6: 'C', 7: 'N', 8: 'O', 
        9: 'F', 10: 'Ne', 11: 'Na', 12: 'Mg', 13: 'Al', 14: 'Si', 15: 'P', 
        16: 'S', 17: 'Cl', 18: 'Ar', 19: 'K', 20: 'Ca', 35: 'Br', 53: 'I'
    }
    
    # Convert Bohr to Angstrom (1 Bohr = 0.529177 Angstrom)
    # bohr_to_angstrom = 0.529177210903
    positions_angstrom = positions
    
    # Build XYZ format
    n_atoms = len(atomic_numbers)
    xyz = f"{n_atoms}\n"
    xyz += "Molecule from DFT calculation\n"
    
    for z, pos in zip(atomic_numbers, positions_angstrom):
        symbol = element_map.get(int(z), f'X{int(z)}')
        xyz += f"{symbol:2s} {pos[0]:12.6f} {pos[1]:12.6f} {pos[2]:12.6f}\n"
    
    return xyz

def view_molecule_3d(atomic_numbers, positions, size=(600, 400), style='stick', surface=False, opacity=0.5):
    """Visualize molecule in 3D using py3Dmol"""
    assert style in ('line', 'stick', 'sphere', 'cartoon')
    
    # Convert to XYZ format
    xyz_data = atom_data_to_xyz(atomic_numbers, positions)
    
    # Create viewer
    viewer = py3Dmol.view(width=size[0], height=size[1])
    viewer.addModel(xyz_data, 'xyz')
    viewer.setStyle({"stick": {"radius": 0.05}, "sphere": {"scale": 0.2}})
    
    if surface:
        viewer.addSurface(py3Dmol.SAS, {'opacity': opacity})
    
    viewer.zoomTo()
    return viewer

# # Visualize the molecule
# viewer = view_molecule_3d(atomic_number, pos, size=(800, 600), style='sphere', opacity=0.5)
# viewer.show()