import json
import os
import csv
from pathlib import Path

def extract_important_values(json_dir, output_file):
    """
    Extract important values from JSON result files and save to a readable text file.
    """
    json_dir = Path(json_dir)
    output_file = Path(output_file)
    
    # Find all JSON files
    json_files = sorted(json_dir.glob("*.json"))
    
    results = []
    
    for json_file in json_files:
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        if 'error' in data:
            continue
        
        filename = data.get('filename', 'N/A')
        atom_count = data.get('atom_count', 0)
        
        # Extract energies
        scflow_e = data.get('scflow', {}).get('energy', {})
        rks_e = data.get('rks', {}).get('energy', {})
        scf_e = data.get('scf', {}).get('energy', {})
        
        # Extract times
        scflow_time = data.get('scflow', {}).get('time', None)
        rks_time = data.get('rks', {}).get('time', None)
        scf_time = data.get('scf', {}).get('time', None)
        scf_cycle = data.get('scf', {}).get('cycle', None)
        
        # Extract differences
        diff_rks_scflow = data.get('differences', {}).get('rks_vs_scflow', {})
        diff_scf_scflow = data.get('differences', {}).get('scf_vs_scflow', {})
        diff_rks_scf = data.get('differences', {}).get('rks_vs_scf', {})
        
        result = {
            'filename': filename,
            'atom_count': atom_count,
            
            # Energies (eV)
            'scflow_energy_eV': scflow_e.get('eV'),
            'rks_energy_eV': rks_e.get('eV'),
            'scf_energy_eV': scf_e.get('eV'),
            
            # Energies (meV)
            'scflow_energy_meV': scflow_e.get('meV'),
            'rks_energy_meV': rks_e.get('meV'),
            'scf_energy_meV': scf_e.get('meV'),
            
            # Energies per atom (eV/atom)
            'scflow_energy_eV_per_atom': scflow_e.get('eV_per_atom'),
            'rks_energy_eV_per_atom': rks_e.get('eV_per_atom'),
            'scf_energy_eV_per_atom': scf_e.get('eV_per_atom'),
            
            # Energies per atom (meV/atom)
            'scflow_energy_meV_per_atom': scflow_e.get('meV_per_atom'),
            'rks_energy_meV_per_atom': rks_e.get('meV_per_atom'),
            'scf_energy_meV_per_atom': scf_e.get('meV_per_atom'),
            
            # Times
            'scflow_time': scflow_time,
            'rks_time': rks_time,
            'scf_time': scf_time,
            'scf_cycle': scf_cycle,
            
            # Energy differences (eV)
            'rks_scflow_diff_eV': diff_rks_scflow.get('energy', {}).get('eV'),
            'scf_scflow_diff_eV': diff_scf_scflow.get('energy', {}).get('eV'),
            'rks_scf_diff_eV': diff_rks_scf.get('energy', {}).get('eV'),
            
            # Energy differences (meV)
            'rks_scflow_diff_meV': diff_rks_scflow.get('energy', {}).get('meV'),
            'scf_scflow_diff_meV': diff_scf_scflow.get('energy', {}).get('meV'),
            'rks_scf_diff_meV': diff_rks_scf.get('energy', {}).get('meV'),
            
            # Energy differences per atom (eV/atom)
            'rks_scflow_diff_eV_per_atom': diff_rks_scflow.get('energy', {}).get('eV_per_atom'),
            'scf_scflow_diff_eV_per_atom': diff_scf_scflow.get('energy', {}).get('eV_per_atom'),
            'rks_scf_diff_eV_per_atom': diff_rks_scf.get('energy', {}).get('eV_per_atom'),
            
            # Energy differences per atom (meV/atom)
            'rks_scflow_diff_meV_per_atom': diff_rks_scflow.get('energy', {}).get('meV_per_atom'),
            'scf_scflow_diff_meV_per_atom': diff_scf_scflow.get('energy', {}).get('meV_per_atom'),
            'rks_scf_diff_meV_per_atom': diff_rks_scf.get('energy', {}).get('meV_per_atom'),
            
            # Forces MAE (eV/Å)
            'rks_scflow_forces_mae_eV_per_ang': diff_rks_scflow.get('forces_mae', {}).get('eV_per_ang'),
            'scf_scflow_forces_mae_eV_per_ang': diff_scf_scflow.get('forces_mae', {}).get('eV_per_ang'),
            'rks_scf_forces_mae_eV_per_ang': diff_rks_scf.get('forces_mae', {}).get('eV_per_ang'),
            
            # Forces MAE (meV/Å)
            'rks_scflow_forces_mae_meV_per_ang': diff_rks_scflow.get('forces_mae', {}).get('meV_per_ang'),
            'scf_scflow_forces_mae_meV_per_ang': diff_scf_scflow.get('forces_mae', {}).get('meV_per_ang'),
            'rks_scf_forces_mae_meV_per_ang': diff_rks_scf.get('forces_mae', {}).get('meV_per_ang'),
        }
        
        results.append(result)
    
    # Write to text file in a readable format
    with open(output_file, 'w') as f:
        f.write("=" * 120 + "\n")
        f.write("DFT Calculation Results Summary\n")
        f.write("=" * 120 + "\n\n")
        
        for i, result in enumerate(results, 1):
            f.write(f"\n{'=' * 120}\n")
            f.write(f"Sample {i}: {result['filename']} ({result['atom_count']} atoms)\n")
            f.write(f"{'=' * 120}\n\n")
            
            # Energies
            f.write("Energies:\n")
            f.write(f"  SCFlow:  {result['scflow_energy_eV']:>15.6f} eV  ({result['scflow_energy_meV']:>15.3f} meV)\n")
            f.write(f"  RKS:     {result['rks_energy_eV']:>15.6f} eV  ({result['rks_energy_meV']:>15.3f} meV)\n")
            if result['scf_energy_eV'] is not None:
                f.write(f"  SCF:     {result['scf_energy_eV']:>15.6f} eV  ({result['scf_energy_meV']:>15.3f} meV)\n")
            f.write("\n")
            
            # Energies per atom
            f.write("Energies per atom:\n")
            f.write(f"  SCFlow:  {result['scflow_energy_eV_per_atom']:>15.6f} eV/atom  ({result['scflow_energy_meV_per_atom']:>15.3f} meV/atom)\n")
            f.write(f"  RKS:     {result['rks_energy_eV_per_atom']:>15.6f} eV/atom  ({result['rks_energy_meV_per_atom']:>15.3f} meV/atom)\n")
            if result['scf_energy_eV_per_atom'] is not None:
                f.write(f"  SCF:     {result['scf_energy_eV_per_atom']:>15.6f} eV/atom  ({result['scf_energy_meV_per_atom']:>15.3f} meV/atom)\n")
            f.write("\n")
            
            # Energy differences
            f.write("Energy Differences:\n")
            if result['rks_scflow_diff_eV'] is not None:
                f.write(f"  RKS - SCFlow:   {result['rks_scflow_diff_eV']:>15.6f} eV  ({result['rks_scflow_diff_meV']:>15.3f} meV)\n")
                f.write(f"                   {result['rks_scflow_diff_eV_per_atom']:>15.6f} eV/atom  ({result['rks_scflow_diff_meV_per_atom']:>15.3f} meV/atom)\n")
            if result['scf_scflow_diff_eV'] is not None:
                f.write(f"  SCF - SCFlow:   {result['scf_scflow_diff_eV']:>15.6f} eV  ({result['scf_scflow_diff_meV']:>15.3f} meV)\n")
                f.write(f"                   {result['scf_scflow_diff_eV_per_atom']:>15.6f} eV/atom  ({result['scf_scflow_diff_meV_per_atom']:>15.3f} meV/atom)\n")
            if result['rks_scf_diff_eV'] is not None:
                f.write(f"  SCF - RKS:      {result['rks_scf_diff_eV']:>15.6f} eV  ({result['rks_scf_diff_meV']:>15.3f} meV)\n")
                f.write(f"                   {result['rks_scf_diff_eV_per_atom']:>15.6f} eV/atom  ({result['rks_scf_diff_meV_per_atom']:>15.3f} meV/atom)\n")
            f.write("\n")
            
            # Forces MAE
            f.write("Forces MAE:\n")
            if result['rks_scflow_forces_mae_eV_per_ang'] is not None:
                f.write(f"  RKS - SCFlow:   {result['rks_scflow_forces_mae_eV_per_ang']:>15.6f} eV/Å  ({result['rks_scflow_forces_mae_meV_per_ang']:>15.3f} meV/Å)\n")
            if result['scf_scflow_forces_mae_eV_per_ang'] is not None:
                f.write(f"  SCF - SCFlow:   {result['scf_scflow_forces_mae_eV_per_ang']:>15.6f} eV/Å  ({result['scf_scflow_forces_mae_meV_per_ang']:>15.3f} meV/Å)\n")
            if result['rks_scf_forces_mae_eV_per_ang'] is not None:
                f.write(f"  SCF - RKS:      {result['rks_scf_forces_mae_eV_per_ang']:>15.6f} eV/Å  ({result['rks_scf_forces_mae_meV_per_ang']:>15.3f} meV/Å)\n")
            f.write("\n")
            
            # Computation times
            f.write("Computation Times:\n")
            f.write(f"  SCFlow:  {result['scflow_time']:>15.4f} s\n")
            f.write(f"  RKS:     {result['rks_time']:>15.4f} s\n")
            if result['scf_time'] is not None:
                f.write(f"  SCF:     {result['scf_time']:>15.4f} s  (cycles: {result['scf_cycle']})\n")
            f.write("\n")
        
        # Write summary statistics
        f.write("\n" + "=" * 120 + "\n")
        f.write("Summary Statistics\n")
        f.write("=" * 120 + "\n\n")
        
        if results:
            # Calculate statistics
            valid_rks_scflow = [r['rks_scflow_diff_meV_per_atom'] for r in results if r['rks_scflow_diff_meV_per_atom'] is not None]
            valid_scf_scflow = [r['scf_scflow_diff_meV_per_atom'] for r in results if r['scf_scflow_diff_meV_per_atom'] is not None]
            valid_rks_scf = [r['rks_scf_diff_meV_per_atom'] for r in results if r['rks_scf_diff_meV_per_atom'] is not None]
            
            valid_forces_rks_scflow = [r['rks_scflow_forces_mae_meV_per_ang'] for r in results if r['rks_scflow_forces_mae_meV_per_ang'] is not None]
            valid_forces_scf_scflow = [r['scf_scflow_forces_mae_meV_per_ang'] for r in results if r['scf_scflow_forces_mae_meV_per_ang'] is not None]
            valid_forces_rks_scf = [r['rks_scf_forces_mae_meV_per_ang'] for r in results if r['rks_scf_forces_mae_meV_per_ang'] is not None]
            
            f.write("Energy Differences per Atom (meV/atom):\n")
            if valid_rks_scflow:
                f.write(f"  RKS - SCFlow:   Mean = {sum(valid_rks_scflow)/len(valid_rks_scflow):>10.3f} meV/atom, "
                       f"MAE = {sum(abs(x) for x in valid_rks_scflow)/len(valid_rks_scflow):>10.3f} meV/atom, "
                       f"RMSE = {(sum(x**2 for x in valid_rks_scflow)/len(valid_rks_scflow))**0.5:>10.3f} meV/atom\n")
            if valid_scf_scflow:
                f.write(f"  SCF - SCFlow:   Mean = {sum(valid_scf_scflow)/len(valid_scf_scflow):>10.3f} meV/atom, "
                       f"MAE = {sum(abs(x) for x in valid_scf_scflow)/len(valid_scf_scflow):>10.3f} meV/atom, "
                       f"RMSE = {(sum(x**2 for x in valid_scf_scflow)/len(valid_scf_scflow))**0.5:>10.3f} meV/atom\n")
            if valid_rks_scf:
                f.write(f"  SCF - RKS:      Mean = {sum(valid_rks_scf)/len(valid_rks_scf):>10.3f} meV/atom, "
                       f"MAE = {sum(abs(x) for x in valid_rks_scf)/len(valid_rks_scf):>10.3f} meV/atom, "
                       f"RMSE = {(sum(x**2 for x in valid_rks_scf)/len(valid_rks_scf))**0.5:>10.3f} meV/atom\n")
            f.write("\n")
            
            f.write("Forces MAE (meV/Å):\n")
            if valid_forces_rks_scflow:
                f.write(f"  RKS - SCFlow:   Mean = {sum(valid_forces_rks_scflow)/len(valid_forces_rks_scflow):>10.3f} meV/Å\n")
            if valid_forces_scf_scflow:
                f.write(f"  SCF - SCFlow:   Mean = {sum(valid_forces_scf_scflow)/len(valid_forces_scf_scflow):>10.3f} meV/Å\n")
            if valid_forces_rks_scf:
                f.write(f"  SCF - RKS:      Mean = {sum(valid_forces_rks_scf)/len(valid_forces_rks_scf):>10.3f} meV/Å\n")
            f.write("\n")
            
            f.write(f"Total samples: {len(results)}\n")
    
    print(f"Summary saved to {output_file}")
    print(f"Processed {len(results)} samples")
    
    # Also save as CSV for easy analysis
    csv_file = output_file.with_suffix('.csv')
    if results:
        fieldnames = [
            'filename', 'atom_count',
            'scflow_energy_eV', 'scflow_energy_meV', 'scflow_energy_eV_per_atom', 'scflow_energy_meV_per_atom',
            'rks_energy_eV', 'rks_energy_meV', 'rks_energy_eV_per_atom', 'rks_energy_meV_per_atom',
            'scf_energy_eV', 'scf_energy_meV', 'scf_energy_eV_per_atom', 'scf_energy_meV_per_atom',
            'scflow_time', 'rks_time', 'scf_time', 'scf_cycle',
            'rks_scflow_diff_eV', 'rks_scflow_diff_meV', 'rks_scflow_diff_eV_per_atom', 'rks_scflow_diff_meV_per_atom',
            'scf_scflow_diff_eV', 'scf_scflow_diff_meV', 'scf_scflow_diff_eV_per_atom', 'scf_scflow_diff_meV_per_atom',
            'rks_scf_diff_eV', 'rks_scf_diff_meV', 'rks_scf_diff_eV_per_atom', 'rks_scf_diff_meV_per_atom',
            'rks_scflow_forces_mae_eV_per_ang', 'rks_scflow_forces_mae_meV_per_ang',
            'scf_scflow_forces_mae_eV_per_ang', 'scf_scflow_forces_mae_meV_per_ang',
            'rks_scf_forces_mae_eV_per_ang', 'rks_scf_forces_mae_meV_per_ang',
        ]
        
        with open(csv_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        
        print(f"CSV summary saved to {csv_file}")

if __name__ == "__main__":
    json_dir = "/root/25DFT/QHFlow/src/md/test_script/dft_test_results"
    output_file = "/root/25DFT/QHFlow/src/md/test_script/dft_test_results_summary.txt"
    
    extract_important_values(json_dir, output_file)

