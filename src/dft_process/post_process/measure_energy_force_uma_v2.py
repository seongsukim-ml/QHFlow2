import sys
import os
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.append(src_path)

from common.custom_logger import get_logger
logger = get_logger(__file__)
logger.info(f"Source path: {src_path}")

from dft_process.dft_process_utils import *
from argparse import ArgumentParser
import multiprocessing as mp
# from tqdm.rich import tqdm
from tqdm import tqdm
import gc
import time
import traceback
import concurrent.futures
from ase.atoms import Atoms
from fairchem.core import pretrained_mlip, FAIRChemCalculator

predictor = pretrained_mlip.get_predict_unit("uma-s-1p1", device="cuda")
uma = FAIRChemCalculator(predictor, task_name="omol")

torch.set_num_threads(4)

def eexx():
    os._exit(0)

def process_single_file(args):
    """Process a single file tuple (pred_path, gt_path, calc_path)"""
    pred_path, gt_path, calc_path, omol_path = args
    # change parent directory to "processed"
    pred_path_parent = os.path.dirname(pred_path)
    pred_path_processed = pred_path.replace(pred_path_parent, f"{pred_path_parent}_processed")

    gt_path_parent = os.path.dirname(os.path.dirname(gt_path))
    gt_path_processed = os.path.join(gt_path_parent, f"gt_processed", os.path.basename(gt_path))
    # gt_path_processed = gt_path_processed.replace(".pt", "_processed.pt")
    try:
        if os.path.exists(pred_path_processed):
            pred_data = torch.load(pred_path_processed)
        else:
            pred_data = torch.load(pred_path)

        if os.path.exists(gt_path_processed):
            gt_data = torch.load(gt_path_processed)
        else:
            gt_data = torch.load(gt_path)
        atoms = gt_data["atoms"]
        pos = gt_data["pos"]
        length_unit = gt_data["length_unit"]
        # _dtype = pred_data["pred_hamiltonian"].dtype
        _dtype = torch.float64
        if length_unit.lower() == "bohr":
            pos = pos * BOHR2ANG 
        elif length_unit.lower() == "angstrom":
            pass
        else:
            raise ValueError(f"Invalid length unit: {length_unit}")

        matrix_format = gt_data["format"]
        if matrix_format.lower() == "pyscf_def2svp":
            convention = None
        elif matrix_format.lower() == "e3nn":
            convention = "e3nn_to_pyscf_def2svp"
        else:
            raise ValueError(f"Invalid matrix format: {matrix_format}")

        calc_mf = init_pyscf_mf(atoms, pos, unit="ang")
        grad_frame = calc_mf.nuc_grad_method()
        # Check if calculated data exists
        if not os.path.exists(calc_path):
            calc_data = gt_data.copy()  # Use copy to avoid modifying original
            start_time = time.time()
            calc_data["calc_energy"] = calc_mf.kernel()
            calc_data["calc_time"] = time.time() - start_time
            calc_data["hamiltonian"] = torch.tensor(calc_mf.get_fock(dm=calc_mf.make_rdm1()), dtype=torch.float64)
            calc_data["overlap"] = torch.tensor(calc_mf.get_ovlp(), dtype=torch.float64)
            calc_data["density_matrix"] = torch.tensor(calc_mf.make_rdm1(), dtype=torch.float64)
            calc_data["method"] = "RKS"
            calc_data["xc"] = "pbe"
            calc_data["basis"] = "def2svp"
            calc_data["scf_cycles"] = calc_mf.cycles
            calc_data["calc_forces"] = torch.tensor(-grad_frame.kernel(), dtype=torch.float64)
            torch.save(calc_data, calc_path)
        else:
            try:
                calc_data = torch.load(calc_path)
            except:
                calc_data = gt_data.copy()  # Use copy to avoid modifying original
                start_time = time.time()
                calc_data["calc_energy"] = calc_mf.kernel()
                calc_data["calc_time"] = time.time() - start_time
                calc_data["hamiltonian"] = torch.tensor(calc_mf.get_fock(dm=calc_mf.make_rdm1()), dtype=torch.float64)
                calc_data["overlap"] = torch.tensor(calc_mf.get_ovlp(), dtype=torch.float64)
                calc_data["density_matrix"] = torch.tensor(calc_mf.make_rdm1(), dtype=torch.float64)
                calc_data["method"] = "RKS"
                calc_data["xc"] = "pbe"
                calc_data["basis"] = "def2svp"
                calc_data["scf_cycles"] = calc_mf.cycles
                calc_data["calc_forces"] = torch.tensor(-grad_frame.kernel(), dtype=torch.float64)
                torch.save(calc_data, calc_path)
        # Calculate density matrices
        calc_overlap = calc_data["overlap"].unsqueeze(0) # (gt_overlap - calc_overlap) has float32 precision error (1e^-7)
        # calc_ham = calc_data["hamiltonian"].unsqueeze(0)
 
        calc_forces = np.array(calc_data["calc_forces"])
        calc_energy = calc_data["calc_energy"]
        
        if "calc_forces" in pred_data:
            pred_energy = pred_data["calc_energy"]
            pred_forces = pred_data["calc_forces"]
            if not os.path.exists(pred_path_processed):
                torch.save(pred_data, pred_path_processed)
        else:
            calc_overlap = calc_data["overlap"].unsqueeze(0).to(_dtype) # (gt_overlap - calc_overlap) has float32 precision error (1e^-7)
            pred_ham = matrix_transform_single(pred_data["pred_hamiltonian"].unsqueeze(0), atoms, convention=convention).to(_dtype)
            
            pred_density, pred_res = calc_dm0_from_ham_(atoms, calc_overlap, pred_ham)
            pred_energy = calc_mf.energy_tot(pred_density)
            pred_data["calc_energy"] = pred_energy

            pred_mo_energy = pred_res["orbital_energies"].squeeze().numpy()
            pred_mo_coeff = pred_res["orbital_coefficients"].squeeze().numpy()

            mo_occ = calc_mf.get_occ(pred_mo_energy, pred_mo_coeff)
            pred_forces = -grad_frame.kernel(mo_energy=pred_mo_energy, mo_coeff=-pred_mo_coeff, mo_occ=mo_occ)
            pred_data["calc_forces"] = pred_forces

            torch.save(pred_data, pred_path_processed)
        
        if "calc_forces" in gt_data:
            gt_energy = gt_data["calc_energy"]
            gt_forces = gt_data["calc_forces"]
            if not os.path.exists(gt_path_processed):
                torch.save(gt_data, gt_path_processed)
        else:
            calc_overlap = calc_data["overlap"].unsqueeze(0).to(_dtype) # (gt_overlap - calc_overlap) has float32 precision error (1e^-7)
            gt_ham = matrix_transform_single(gt_data["hamiltonian"].unsqueeze(0), atoms, convention=convention).to(_dtype)
            
            gt_density, gt_res = calc_dm0_from_ham_(atoms, calc_overlap, gt_ham)
            gt_energy = calc_mf.energy_tot(gt_density)
            gt_data["calc_energy"] = gt_energy

            gt_mo_energy = gt_res["orbital_energies"].squeeze().numpy()
            gt_mo_coeff = gt_res["orbital_coefficients"].squeeze().numpy()

            mo_occ = calc_mf.get_occ(gt_mo_energy, gt_mo_coeff)
            gt_forces = -grad_frame.kernel(mo_energy=gt_mo_energy, mo_coeff=-gt_mo_coeff, mo_occ=mo_occ)
            gt_data["calc_forces"] = gt_forces
            
            torch.save(gt_data, gt_path_processed)
        
        # gt_data_energy = gt_data["energy"].item()

        pred_forces_norm = np.linalg.norm(pred_forces, axis=1)
        gt_forces_norm = np.linalg.norm(gt_forces, axis=1)
        calc_forces_norm = np.linalg.norm(calc_forces, axis=1)
        
        res = {
                "pred_energy": pred_energy,
                "gt_energy": gt_energy,
                "calc_energy": calc_energy,

                "energy_diff (pred-gt)": pred_energy - gt_energy,
                "energy_diff (pred-calc_energy)": pred_energy - calc_energy,
                "energy_diff (gt-calc_energy)": gt_energy - calc_energy,

                "pred_force": pred_forces,
                "gt_force": gt_forces,
                "calc_force": calc_forces,

                "forces_diff l2 (pred-gt)": abs(pred_forces - gt_forces).mean(),
                "forces_diff l2 (pred-calc_forces)": abs(pred_forces - calc_forces).mean(),
                "forces_diff l2 (gt-calc_forces)": abs(gt_forces - calc_forces).mean(),

                "pred_force_norm": pred_forces_norm,
                "gt_force_norm": gt_forces_norm,
                "calc_force_norm": calc_forces_norm,

                "pred_force_norm_diff (pred-gt)": abs(pred_forces_norm - gt_forces_norm).mean(),
                "pred_force_norm_diff (pred-calc_forces)": abs(pred_forces_norm - calc_forces_norm).mean(),
                "pred_force_norm_diff (gt-calc_forces)": abs(gt_forces_norm - calc_forces_norm).mean()
            }
        if omol_path is None:
            # Clean up memory
            return res
        
        else:
            if os.path.exists(omol_path):
                omol_data = torch.load(omol_path)
                omol_energy = omol_data["energy"] # eV
                omol_forces = omol_data["forces"] # eV/Ang
            else:
                start_time = time.time()
                uma_atom = Atoms(numbers=atoms.squeeze(), positions=pos)
                uma_atom.info["charge"] = 0
                uma_atom.info["spin"] = 0
                uma.calculate(uma_atom, ["energy, forces"], [])
                uma_time = time.time() - start_time
                omol_energy = uma.results["energy"] # eV
                omol_forces = uma.results["forces"] # eV/Ang
                omol_data = {"energy": omol_energy, "forces": omol_forces, "pos": pos, "atoms": atoms, "uma_time": uma_time}
                torch.save(omol_data, omol_path)

            omol_energy = omol_energy * eV2HA
            omol_forces = omol_forces.astype(calc_forces.dtype) * eV2HA / ANG2BOHR
            omol_forces_norm = np.linalg.norm(omol_forces, axis=1)
            
            res2 = {
                "omol_energy": omol_energy,

                "energy_diff (gt-omol_energy)": gt_energy - omol_energy,
                "energy_diff (calc-omol_energy)": calc_energy - omol_energy,

                "omol_forces": omol_forces,

                "forces_diff l2 (gt-omol_forces)": abs(gt_forces - omol_forces).mean(),
                "forces_diff l2 (calc-omol_forces)": abs(calc_forces - omol_forces).mean(),

                "omol_force_norm": omol_forces_norm,

                "gt_force_norm_diff (gt-omol_forces)": abs(gt_forces_norm - omol_forces_norm).mean(),
                "gt_force_norm_diff (calc-omol_forces)": abs(calc_forces_norm - omol_forces_norm).mean(),
            }      
            return {**res, **res2}
        
    except Exception as e:
        # import pdb; pdb.set_trace()
        error_msg = f"Error processing {pred_path}: {str(e)}\nTraceback:\n{traceback.format_exc()}"
        logger.error(error_msg)
        error_dir = os.path.dirname(os.path.dirname(pred_path))
        error_file_path = os.path.join(error_dir, "error_files.txt")
        logger.error(f"Error file path: {error_file_path}")
        with open(error_file_path, "a") as f:
            f.write(f"{pred_path} {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

        return {"error": error_msg, "file": pred_path, "line_number": traceback.extract_tb(e.__traceback__)[-1].lineno}


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
    logger.info(f"  Processing {len(file_batch)} files in chunks of {chunk_size} with {num_workers} workers")
    
    all_results = []
    total_chunks = (len(file_batch) + chunk_size - 1) // chunk_size
    
    for i in range(0, len(file_batch), chunk_size):
        chunk = file_batch[i:i + chunk_size]
        chunk_num = i // chunk_size + 1
        
        logger.info(f"  Processing chunk {chunk_num}/{total_chunks} ({len(chunk)} files)")
        
        try:
            chunk_results = process_batch_stable(chunk, num_workers, timeout)
            all_results.extend(chunk_results)
            
            # Small delay between chunks to prevent resource exhaustion
            time.sleep(1)
            
        except Exception as e:
            logger.error(f"  Error processing chunk {chunk_num}: {str(e)}")
            # Continue with next chunk
            continue
    
    logger.info(f"  Total completed: {len(all_results)} successful")
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
                logger.warning(f"  Skipping file due to error: {result['error']}")
        except Exception as e:
            logger.error(f"  Error processing file {i+1}/{len(file_tuples)}: {str(e)}")
            continue
    
    return results


def pandas_summary(results, output_path, postfix, keys, unit="meV"):
    """Summary statistics for pandas DataFrame"""
    summary_path = output_path.replace(".csv", f"_summary_{postfix}.csv")
    
    # Default units
    # E energy in Hartree
    # F force in Hartree/Bohr
    # D distance in Angstrom

    unit_conversion = {
        "meV".lower(): HA2meV,
        "mu_hartree".lower(): 1e6,
        "meV/Angstrom".lower(): HA_BOHR_2_meV_ANG,
        "meV/Ang".lower(): HA_BOHR_2_meV_ANG,
        "mu_hartree/Bohr".lower(): 1e6
    }
    unit = unit.lower()
    assert unit in unit_conversion, f"Invalid unit: {unit}, available units: {unit_conversion.keys()}"
    
    columns_names = []
    values = []
    for key in keys:
        columns_names.append(f"mean {key} {unit}")
        values.append(results[key].mean() * unit_conversion[unit])
    for key in keys:
        columns_names.append(f"std  {key} {unit}")
        values.append(results[key].std() * unit_conversion[unit])
    
    summary_results = pd.DataFrame([],
    columns=columns_names)
    summary_results.loc[0] = values
    summary_results.to_csv(summary_path, index=False)
    logger.info(f"Summary results saved to: {summary_path}")

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
    reverse_order=False,
    uma=False
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
        logger.info(f"CPU usage limited to: {max_cpu_cores} cores")
    else:
        max_cpu_cores = cpu_count
    
    # Ensure num_workers doesn't exceed the CPU limit
    actual_workers = min(num_workers, max_cpu_cores)
    logger.info(f"Using: {actual_workers} workers (capped at {max_cpu_cores} cores)")
    
    # Initialize experiment
    model_path_cls = ModelPath(
        dataset_name=dataset_name,
        model_name=model_name,
        model_prefix=model_prefix,
        model_postfix=model_postfix
    )
    
    # Get file lists
    if uma:
        file_name = f"energy_force_uma_results_{dataset_name}_{model_name}.csv"
    else:
        file_name = f"energy_force_results_{dataset_name}_{model_name}.csv"
    
    output_path = model_path_cls.model_path + f"/{file_name}"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    pred_list = model_path_cls.pred_list()
    gt_list = model_path_cls.gt_list()
    calc_list = model_path_cls.calc_list()

    pred_processed_dir = os.path.dirname(pred_list[0]).replace(os.path.dirname(pred_list[0]), f"{os.path.dirname(pred_list[0])}_processed")
    gt_processed_dir = os.path.dirname(gt_list[0]).replace(os.path.dirname(gt_list[0]), f"{os.path.dirname(gt_list[0])}_processed")
    os.makedirs(pred_processed_dir, exist_ok=True)
    os.makedirs(gt_processed_dir, exist_ok=True)

    if uma:
        omol_list = model_path_cls.custom_list(path=model_path_cls.custom_path("omol_calc"), path_name="omol")
    else:
        omol_list = [None] * len(pred_list)

    if reverse_order:
        pred_list = pred_list[::-1]
        gt_list = gt_list[::-1]
        calc_list = calc_list[::-1]
        omol_list = omol_list[::-1]
    
    total_files = len(pred_list)
    logger.info(f"Total files to process: {total_files}")
    logger.info(f"Start fraction: {start_frac}, End fraction: {end_frac}")
    start_index = int(total_files * start_frac)
    end_index = int(total_files * end_frac)
    sliced_total_files = end_index - start_index
    logger.info(f"Start index: {start_index}, End index: {end_index}")
    logger.info(f"Total files to process: {sliced_total_files}")
    
    
    sliced_pred_list = pred_list[start_index:end_index]
    sliced_gt_list = gt_list[start_index:end_index]
    sliced_calc_list = calc_list[start_index:end_index]
    sliced_omol_list = omol_list[start_index:end_index]
    # Create file tuples
    file_tuples = list(zip(sliced_pred_list, sliced_gt_list, sliced_calc_list, sliced_omol_list))
    
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
            
            energy_keys = [
                "energy_diff (pred-gt)",
                "energy_diff (pred-calc_energy)",
                "energy_diff (gt-calc_energy)",
            ]
            if uma:
                energy_keys.extend([
                    "energy_diff (gt-omol_energy)",
                    "energy_diff (calc-omol_energy)",
                ])
            
            pandas_summary(
                results,
                output_path,
                postfix="energy_diff_meV",
                keys=energy_keys,
                unit="meV")

            pandas_summary(
                results,
                output_path,
                postfix="energy_diff_mu_hartree",
                keys=energy_keys,
                unit="mu_hartree")
            
            force_keys = [
                "forces_diff l2 (pred-gt)",
                "forces_diff l2 (pred-calc_forces)",
                "forces_diff l2 (gt-calc_forces)",                
            ]
            if uma:
                force_keys.extend([
                    "forces_diff l2 (gt-omol_forces)",
                    "forces_diff l2 (calc-omol_forces)",
                ])
            
            pandas_summary(
                results,
                output_path,
                postfix="force_diff_meV_Ang",
                keys=force_keys,
                unit="meV/Ang")

            pandas_summary(
                results,
                output_path,
                "force_diff_mu_hartree_Bohr",
                keys=force_keys,
                unit="mu_hartree/Bohr")             
             
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
    parser.add_argument("--uma", action="store_true", help="Use Uma")
    args = parser.parse_args()
    
    main(args.dataset_name, args.model_name, args.model_prefix, args.model_postfix,
         batch_size=args.batch_size, use_parallel=not args.no_parallel, 
         num_workers=args.num_workers, max_cpu_cores=args.max_cpu_cores,
         start_frac=args.start_frac, end_frac=args.end_frac,
         reverse_order=args.reverse_order,
         uma=args.uma)