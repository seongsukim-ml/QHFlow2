import sys
import os
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.append(src_path)

from qhflow2.common.custom_logger import get_logger
logger = get_logger(__file__)
logger.info(f"Source path: {src_path}")

from qhflow2.dft_process.dft_process_utils import *

import pyscf
from argparse import ArgumentParser
import multiprocessing as mp
from tqdm import tqdm
import gc
import time
import os
import logging
import traceback
import concurrent.futures
from ase.build import molecule
from ase.atoms import Atoms
from fairchem.core import pretrained_mlip, FAIRChemCalculator

predictor = pretrained_mlip.get_predict_unit("uma-s-1p1", device="cuda")
uma = FAIRChemCalculator(predictor, task_name="omol")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)
torch.set_num_threads(8)

def process_single_file(args):
    """Process a single file tuple (pred_path, gt_path, calc_path)"""
    gt_path, calc_path, omol_path = args
    
    try:
        gt_data = torch.load(gt_path)
        atoms = gt_data["atoms"]
        pos = gt_data["pos"] * BOHR2ANG

        calc_mf = init_pyscf_mf(atoms, pos, unit="ang")
        grad_frame = calc_mf.nuc_grad_method()
        # Check if calculated data exists
        if not os.path.exists(calc_path):
            calc_data = gt_data.copy()  # Use copy to avoid modifying original
            start_time = time.time()
            calc_mf.kernel()
            calc_data["calc_time"] = time.time() - start_time
            calc_data["hamiltonian"] = torch.tensor(calc_mf.get_fock(dm=calc_mf.make_rdm1()), dtype=torch.float64)
            calc_data["overlap"] = torch.tensor(calc_mf.get_ovlp(), dtype=torch.float64)
            calc_data["density_matrix"] = torch.tensor(calc_mf.make_rdm1(), dtype=torch.float64)
            calc_data["method"] = "RKS"
            calc_data["xc"] = "pbe"
            calc_data["basis"] = "def2svp"
            calc_data["scf_cycles"] = calc_mf.cycles
            calc_data["forces"] = torch.tensor(-grad_frame.kernel(), dtype=torch.float64)
            torch.save(calc_data, calc_path)
        else:
            calc_data = torch.load(calc_path)

        # Calculate density matrices
        # calc_data["overlap"] and calc_data["hamiltonian"] are already transformed
        # pred_ham = matrix_transform_single(pred_data["pred_hamiltonian"].unsqueeze(0), atoms, convention="back2pyscf")
        # gt_ham = matrix_transform_single(gt_data["hamiltonian"].unsqueeze(0), atoms, convention="back2pyscf")
        # gt_overlap = matrix_transform_single(gt_data["overlap"].unsqueeze(0), atoms, convention="back2pyscf")
        # Forces unit: Eh/Bohr
        
        if "calc_forces" in calc_data:
            calc_energy = calc_data["calc_energy"]
            calc_forces = calc_data["calc_forces"]
        else:
            calc_overlap = calc_data["overlap"].unsqueeze(0) # (gt_overlap - calc_overlap) has float32 precision error (1e^-7)
            calc_ham = calc_data["hamiltonian"].unsqueeze(0)
            
            calc_density, calc_res = calc_dm0_from_ham(atoms, calc_overlap, calc_ham, transform=False)
            calc_energy = calc_mf.energy_tot(calc_density)
            calc_data["calc_energy"] = calc_energy
            
            calc_mo_energy = calc_res["orbital_energies"].squeeze().numpy()
            calc_mo_coeff = calc_res["orbital_coefficients"].squeeze().numpy()

            mo_occ = calc_mf.get_occ(calc_mo_energy, calc_mo_coeff)
            calc_forces = -grad_frame.kernel(mo_energy=calc_mo_energy, mo_coeff=calc_mo_coeff, mo_occ=mo_occ)
            calc_data["calc_forces"] = calc_forces
            
            torch.save(calc_data, calc_path)
       
        if os.path.exists(omol_path):
            omol_data = torch.load(omol_path)
            omol_energy = omol_data["energy"]
            omol_forces = omol_data["forces"]
        else:
            uma_atom = Atoms(numbers=atoms, positions=pos)
            uma_atom.info["charge"] = 0
            uma_atom.info["spin"] = 0
            uma.calculate(uma_atom, ["energy, forces"], [])
            omol_energy = uma.results["energy"] # eV
            omol_forces = uma.results["forces"] # eV/Ang
            omol_data = {"energy": omol_energy, "forces": omol_forces, "pos": pos, "atoms": atoms}
            torch.save(omol_data, omol_path)
            
        omol_energy = omol_energy * eV2HA
        omol_forces = omol_forces * eV2HA / ANG2BOHR

        if "calc_forces" in gt_data:
            gt_energy = gt_data["calc_energy"]
            gt_forces = gt_data["calc_forces"]
        else:
            calc_overlap = calc_data["overlap"].unsqueeze(0) # (gt_overlap - calc_overlap) has float32 precision error (1e^-7)
            gt_ham = matrix_transform_single(gt_data["hamiltonian"].unsqueeze(0), atoms, convention="back2pyscf")
            
            gt_density, gt_res = calc_dm0_from_ham(atoms, calc_overlap, gt_ham, transform=False)
            gt_energy = calc_mf.energy_tot(gt_density)
            gt_data["calc_energy"] = gt_energy

            gt_mo_energy = gt_res["orbital_energies"].squeeze().numpy()
            gt_mo_coeff = gt_res["orbital_coefficients"].squeeze().numpy()

            mo_occ = calc_mf.get_occ(gt_mo_energy, gt_mo_coeff)
            gt_forces = -grad_frame.kernel(mo_energy=gt_mo_energy, mo_coeff=-gt_mo_coeff, mo_occ=mo_occ)
            gt_data["calc_forces"] = gt_forces

            torch.save(gt_data, gt_path)

        calc_forces_norm = np.linalg.norm(calc_forces, axis=1)
        gt_forces_norm = np.linalg.norm(gt_forces, axis=1)
        omol_forces_norm = np.linalg.norm(omol_forces, axis=1)

        # Clean up memory
        del gt_data, calc_data, omol_data
        gc.collect()

        return {
            "gt_energy": gt_energy,
            "calc_energy": calc_energy,
            "omol_energy": omol_energy,
            
            "energy_diff (gt-calc_energy)": gt_energy - calc_energy,
            "energy_diff (gt-omol_energy)": gt_energy - omol_energy,
            "energy_diff (calc-omol_energy)": calc_energy - omol_energy,

            "gt_force": gt_forces,
            "calc_force": calc_forces,
            "omol_force": omol_forces,

            "forces_diff l2 (gt-calc_forces)": abs(gt_forces - calc_forces).mean(),
            "forces_diff l2 (gt-omol_forces)": abs(gt_forces - omol_forces).mean(),
            "forces_diff l2 (calc-omol_forces)": abs(calc_forces - omol_forces).mean(),

            "gt_force_norm": gt_forces_norm,
            "calc_force_norm": calc_forces_norm,
            "omol_force_norm": omol_forces_norm,

            "gt_force_norm_diff (gt-calc_forces)": abs(gt_forces_norm - calc_forces_norm).mean(),
            "gt_force_norm_diff (gt-omol_forces)": abs(gt_forces_norm - omol_forces_norm).mean(),
            "gt_force_norm_diff (calc-omol_forces)": abs(calc_forces_norm - omol_forces_norm).mean(),
        }
        
    except Exception as e:
        error_msg = f"Error processing {omol_path}: {str(e)}"
        logger.error(error_msg)
        return {"error": error_msg, "file": omol_path}


def process_batch_stable(file_batch, num_workers=4, timeout=300):
    """Process a batch of files using ThreadPoolExecutor with progress bar"""
    logger.info(f"  Processing batch of {len(file_batch)} files with {num_workers} workers")
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        results = list(
            tqdm(
                executor.map(process_single_file, file_batch),
                total=len(file_batch),
                desc="Processing files"
            )
        )
    
    # Filter out error results
    successful_results = [r for r in results if r and "error" not in r]
    error_results = [r for r in results if r and "error" in r]
    
    logger.info(f"  Completed: {len(successful_results)} successful, {len(error_results)} failed")
    
    if error_results:
        logger.info("  Failed files:")
        for result in error_results[:5]:  # Show first 5 failures
            logger.info(f"    {result.get('file', 'Unknown')}: {result.get('error', 'Unknown error')}")
        if len(error_results) > 5:
            logger.info(f"    ... and {len(error_results) - 5} more")
    
    return successful_results


def process_batch_fast(file_batch, num_workers=4, timeout=300):
    """Alias for process_batch_stable for backward compatibility"""
    return process_batch_stable(file_batch, num_workers, timeout)


def process_batch_chunked(file_batch, num_workers=4, chunk_size=10, timeout=300):
    """Process files in smaller chunks to avoid memory issues"""
    print(f"  Processing {len(file_batch)} files in chunks of {chunk_size} with {num_workers} workers")
    
    all_results = []
    total_chunks = (len(file_batch) + chunk_size - 1) // chunk_size
    
    for i in range(0, len(file_batch), chunk_size):
        chunk = file_batch[i:i + chunk_size]
        chunk_num = i // chunk_size + 1
        
        print(f"  Processing chunk {chunk_num}/{total_chunks} ({len(chunk)} files)")
        
        try:
            chunk_results = process_batch_stable(chunk, num_workers, timeout)
            all_results.extend(chunk_results)
            
            # Small delay between chunks to prevent resource exhaustion
            time.sleep(1)
            
        except Exception as e:
            print(f"  Error processing chunk {chunk_num}: {str(e)}")
            # Continue with next chunk
            continue
    
    print(f"  Total completed: {len(all_results)} successful")
    return all_results


def process_sequential_safe(file_tuples, desc="Processing files"):
    """Process files sequentially with better error handling"""
    results = []
    
    for i, file_tuple in enumerate(tqdm(file_tuples, desc=desc)):
        try:
            result = process_single_file(file_tuple)
            if result and "error" not in result:
                results.append(result)
            elif result and "error" in result:
                print(f"  Skipping file due to error: {result['error']}")
        except Exception as e:
            logger.error(f"  Error processing file {i+1}/{len(file_tuples)}: {str(e)}")
            continue
    
    return results


def main(
    dataset_name,
    model_name,
    model_prefix,
    model_postfix,
    batch_size=10,
    use_parallel=True,
    num_workers=4,
    max_cpu_cores=None,
    start_frac=0.0,
    end_frac=1.0,
    reverse_order=False
    ):
    logger.info(f"Starting energy measurement for dataset: {dataset_name}")
    logger.info(f"Model: {model_name}, Prefix: {model_prefix}, Postfix: {model_postfix}")
    logger.info(f"Configuration: batch_size={batch_size}, workers={num_workers}, parallel={use_parallel}, start_frac={start_frac}, end_frac={end_frac}")
    
    # Show system information and limit CPU usage
    cpu_count = mp.cpu_count()
    logger.info(f"System: {cpu_count} CPU cores available")
    
    # Limit CPU core usage
    if max_cpu_cores is not None:
        max_cpu_cores = min(max_cpu_cores, cpu_count)
        print(f"CPU usage limited to: {max_cpu_cores} cores")
    else:
        max_cpu_cores = cpu_count
    
    # Ensure num_workers doesn't exceed the CPU limit
    actual_workers = min(num_workers, max_cpu_cores)
    logger.info(f"Using: {actual_workers} workers (capped at {max_cpu_cores} cores)")
    
    # Initialize experiment
    md17_experiment = ModelPath(
        dataset_name=dataset_name,
        model_name=model_name,
        model_prefix=model_prefix,
        model_postfix=model_postfix
    )
    
    # Get file lists
    output_path = md17_experiment.custom_path("uma_summary") + f"/energy_force_results_{dataset_name}_{model_name}.csv"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    gt_list = md17_experiment.gt_list()
    calc_list = md17_experiment.calc_list()
    omol_list = md17_experiment.custom_list(path=md17_experiment.custom_path("omol_calc"), path_name="omol")
    if reverse_order:
        gt_list = gt_list[::-1]
        calc_list = calc_list[::-1]
        omol_list = omol_list[::-1]

    total_files = len(gt_list)
    logger.info(f"Total files to process: {total_files}")
    logger.info(f"Start fraction: {start_frac}, End fraction: {end_frac}")
    start_index = int(total_files * start_frac)
    end_index = int(total_files * end_frac)
    sliced_total_files = end_index - start_index
    logger.info(f"Start index: {start_index}, End index: {end_index}")
    logger.info(f"Total files to process: {sliced_total_files}")
    
    
    sliced_gt_list = gt_list[start_index:end_index]
    sliced_calc_list = calc_list[start_index:end_index]
    sliced_omol_list = omol_list[start_index:end_index]
    # Create file tuples
    file_tuples = list(zip(sliced_gt_list, sliced_calc_list, sliced_omol_list))
    
    # Initialize results list
    results_list = []
    
    if os.path.exists(output_path):
        logger.info(f"Results file already exists: {output_path}")
        logger.info(f"Convert all columns to number if it is torch or numpy")
        results_list = pd.read_csv(output_path)
        # convert all columns to number if it is torch or numpy
        results_list = results_list.applymap(lambda x: x.item() if isinstance(x, (torch.Tensor, np.ndarray)) else x)
        results_list = results_list.to_dict(orient="records")
    else:
        if use_parallel and sliced_total_files > batch_size and actual_workers > 1:
            # Process in batches with stable multiprocessing
            logger.info(f"Using parallel processing with batch size: {batch_size}")
            
            for i in tqdm(range(0, total_files, batch_size), desc="Processing batches"):
                batch = file_tuples[i:i + batch_size]
                batch_results = process_batch_stable(batch, actual_workers, timeout=300)
                results_list.extend(batch_results)
                
                # Progress update
                processed = min(i + batch_size, total_files)
                logger.info(f"Processed {processed}/{total_files} files ({processed/total_files*100:.1f}%)")
                
                # Small delay between batches to prevent system overload
                time.sleep(1)
                
        else:
            # Sequential processing with progress bar
            logger.info("Using sequential processing")
            results_list = process_sequential_safe(file_tuples, "Processing files")
        
    # Convert results to DataFrame
    if results_list:
        # Filter out error results
        valid_results = [r for r in results_list if "error" not in r]
        error_count = len(results_list) - len(valid_results)
        
        if valid_results:
            results = pd.DataFrame(valid_results)
            
            # Save results
            output_file = output_path
            results.to_csv(output_file, index=False)
            logger.info(f"Results saved to: {output_file}")
            
            # Print summary statistics
            logger.info("\n=== Summary Statistics ===")
            logger.info(f"Successfully processed: {len(valid_results)}/{total_files} files")
            if error_count > 0:
                logger.info(f"Failed files: {error_count}")
            
            # Energy difference statistics
            # Save summary statistics and energy difference statistics to a csv file
            summary_path = output_path.replace(".csv", "_summary.csv")

            summary_results = pd.DataFrame([],
            columns=[
                "mean energy_diff (gt-calc_energy) mev",
                "mean energy_diff (gt-omol_energy) mev",
                "mean energy_diff (calc-omol_energy) mev",

                "std  energy_diff (gt-calc_energy) mev",
                "std  energy_diff (gt-omol_energy) mev",
                "std  energy_diff (calc-omol_energy) mev",
                
                'mean gt energy meV',
                'std  gt energy meV',

                'mean calc energy meV',
                'std  calc energy meV',

                'mean omol energy meV',
                'std  omol energy meV'
                ])
            
            summary_results.loc[0] = [
                results["energy_diff (gt-calc_energy)"].mean() * HA2meV,
                results["energy_diff (gt-omol_energy)"].mean() * HA2meV,
                results["energy_diff (calc-omol_energy)"].mean() * HA2meV,

                results["energy_diff (gt-calc_energy)"].std() * HA2meV,
                results["energy_diff (gt-omol_energy)"].std() * HA2meV,
                results["energy_diff (calc-omol_energy)"].std() * HA2meV,

                results["gt_energy"].mean() * HA2meV,
                results["gt_energy"].std() * HA2meV,

                results["calc_energy"].mean() * HA2meV,
                results["calc_energy"].std() * HA2meV,

                results["omol_energy"].mean() * HA2meV,
                results["omol_energy"].std() * HA2meV,
            ]

            summary_results.to_csv(summary_path, index=False)

            summary_path_hartree = output_path.replace(".csv", "_summary_mu_hartree.csv")
            summary_results_hartree = pd.DataFrame([],
            columns=[
                "mean energy_diff (gt-calc_energy) hartree",
                "mean energy_diff (gt-omol_energy) hartree",
                "mean energy_diff (calc-omol_energy) hartree",

                "std  energy_diff (gt-calc_energy) hartree",
                "std  energy_diff (gt-omol_energy) hartree",
                "std  energy_diff (calc-omol_energy) hartree",

                "mean gt energy hartree",
                "std  gt energy hartree",

                "mean calc energy hartree",
                "std  calc energy hartree",

                "mean omol energy hartree",
                "std  omol energy hartree",
            ])
            
            summary_results_hartree.loc[0] = [
                results["energy_diff (gt-calc_energy)"].mean() * 1e6,
                results["energy_diff (gt-omol_energy)"].mean() * 1e6,
                results["energy_diff (calc-omol_energy)"].mean() * 1e6,

                results["energy_diff (gt-calc_energy)"].std() * 1e6,
                results["energy_diff (gt-omol_energy)"].std() * 1e6,
                results["energy_diff (calc-omol_energy)"].std() * 1e6,

                results["gt_energy"].mean() * 1e6,
                results["gt_energy"].std() * 1e6,

                results["calc_energy"].mean() * 1e6,
                results["calc_energy"].std() * 1e6,

                results["omol_energy"].mean() * 1e6,
                results["omol_energy"].std() * 1e6,
            ]
            
            summary_results_hartree.to_csv(summary_path_hartree, index=False)

            summary_path_force = output_path.replace(".csv", "_summary_force.csv")
            summary_results_force = pd.DataFrame([],
            columns=[
                "mean forces_diff l2 (gt-calc_forces) meV/Ang",
                "mean forces_diff l2 (gt-omol_forces) meV/Ang",
                "mean forces_diff l2 (calc-omol_forces) meV/Ang",
                
                "mean gt_force_norm_diff (gt-calc_forces) meV/Ang",
                "mean gt_force_norm_diff (gt-omol_forces) meV/Ang",
                "mean gt_force_norm_diff (calc-omol_forces) meV/Ang",
                
                "max forces_diff l2 (gt-calc_forces) meV/Ang",
                "max forces_diff l2 (gt-omol_forces) meV/Ang",
                "max forces_diff l2 (calc-omol_forces) meV/Ang",
                
                "max gt_force_norm_diff (gt-calc_forces) meV/Ang",
                "max gt_force_norm_diff (gt-omol_forces) meV/Ang",
                "max gt_force_norm_diff (calc-omol_forces) meV/Ang",

            ])

            summary_results_force.loc[0] = [
                results["forces_diff l2 (gt-calc_forces)"].mean() * HA_BOHR_2_meV_ANG,
                results["forces_diff l2 (gt-omol_forces)"].mean() * HA_BOHR_2_meV_ANG,
                results["forces_diff l2 (calc-omol_forces)"].mean() * HA_BOHR_2_meV_ANG,

                results["gt_force_norm_diff (gt-calc_forces)"].mean() * HA_BOHR_2_meV_ANG,
                results["gt_force_norm_diff (gt-omol_forces)"].mean() * HA_BOHR_2_meV_ANG,
                results["gt_force_norm_diff (calc-omol_forces)"].mean() * HA_BOHR_2_meV_ANG,

                results["forces_diff l2 (gt-calc_forces)"].max() * HA_BOHR_2_meV_ANG,
                results["forces_diff l2 (gt-omol_forces)"].max() * HA_BOHR_2_meV_ANG,
                results["forces_diff l2 (calc-omol_forces)"].max() * HA_BOHR_2_meV_ANG,

                results["gt_force_norm_diff (gt-calc_forces)"].max() * HA_BOHR_2_meV_ANG,
                results["gt_force_norm_diff (gt-omol_forces)"].max() * HA_BOHR_2_meV_ANG,
                results["gt_force_norm_diff (calc-omol_forces)"].max() * HA_BOHR_2_meV_ANG,
            ]

            summary_results_force.to_csv(summary_path_force, index=False)

            summary_path_force_norm = output_path.replace(".csv", "_summary_force_mu_hartree_bohr.csv")
            summary_results_force_norm = pd.DataFrame([],
            columns=[
                "mean forces_diff l2 (gt-calc_forces) hartree/Bohr",
                "mean forces_diff l2 (gt-omol_forces) hartree/Bohr",
                "mean forces_diff l2 (calc-omol_forces) hartree/Bohr",

                "mean gt_force_norm_diff (gt-calc_forces) hartree/Bohr",
                "mean gt_force_norm_diff (gt-omol_forces) hartree/Bohr",
                "mean gt_force_norm_diff (calc-omol_forces) hartree/Bohr",

                "max forces_diff l2 (gt-calc_forces) hartree/Bohr",
                "max forces_diff l2 (gt-omol_forces) hartree/Bohr",
                "max forces_diff l2 (calc-omol_forces) hartree/Bohr",

                "max gt_force_norm_diff (gt-calc_forces) hartree/Bohr",
                "max gt_force_norm_diff (gt-omol_forces) hartree/Bohr",
                "max gt_force_norm_diff (calc-omol_forces) hartree/Bohr",
            ])
        
            summary_results_force_norm.loc[0] = [
                results["forces_diff l2 (gt-calc_forces)"].mean() * 1e6,
                results["forces_diff l2 (gt-omol_forces)"].mean() * 1e6,
                results["forces_diff l2 (calc-omol_forces)"].mean() * 1e6,

                results["gt_force_norm_diff (gt-calc_forces)"].mean() * 1e6,
                results["gt_force_norm_diff (gt-omol_forces)"].mean() * 1e6,
                results["gt_force_norm_diff (calc-omol_forces)"].mean() * 1e6,

                results["forces_diff l2 (gt-calc_forces)"].max() * 1e6,
                results["forces_diff l2 (gt-omol_forces)"].max() * 1e6,
                results["forces_diff l2 (calc-omol_forces)"].max() * 1e6,

                results["gt_force_norm_diff (gt-calc_forces)"].max() * 1e6,
                results["gt_force_norm_diff (gt-omol_forces)"].max() * 1e6,
                results["gt_force_norm_diff (calc-omol_forces)"].max() * 1e6,
            ]

            summary_results_force_norm.to_csv(summary_path_force_norm, index=False)

            logger.info(f"Summary statistics saved to: {summary_path}")

            logger.info("\n=== Energy Difference Statistics ===")

            logger.info(summary_results)                
            return results
        else:
            logger.info("No valid results generated. All files failed processing.")
            return None
    else:
        logger.info("No results generated. Check for errors in processing.")
        return None


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="QHFlow")
    parser.add_argument("--model_prefix", type=str, default="")
    parser.add_argument("--model_postfix", type=str, default="")
    parser.add_argument("--batch_size", type=int, default=10, 
                       help="Batch size for parallel processing")
    parser.add_argument("--num_workers", type=int, default=4,
                       help="Number of parallel workers (default: 4)")
    parser.add_argument("--max_cpu_cores", type=int, default=None,
                       help="Maximum number of CPU cores to use (default: all available)")
    parser.add_argument("--no_parallel", action="store_true",
                       help="Disable parallel processing")
    parser.add_argument("--start_frac", type=float, default=0.0,
                       help="Fraction of files to start from")
    parser.add_argument("--end_frac", type=float, default=1.0,
                       help="Fraction of files to end at")
    parser.add_argument("--reverse_order", action="store_true",
                       help="Reverse the order of files")
    args = parser.parse_args()
    
    main(args.dataset_name, args.model_name, args.model_prefix, args.model_postfix,
         batch_size=args.batch_size, use_parallel=not args.no_parallel, 
         num_workers=args.num_workers, max_cpu_cores=args.max_cpu_cores,
         start_frac=args.start_frac, end_frac=args.end_frac,
         reverse_order=args.reverse_order)