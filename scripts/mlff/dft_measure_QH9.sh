mamba activate uma
cd ~/25DFT/QHFlow/src/dft_process/post_process

mamba activate pyscf-gpu
cd ~/25DFT/QHFlow/src/dft_process/post_process


# huggingface token
huggingface-cli login
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
########################################################
# QH9Dynamic-300k-geometry QHFlow_so2_v5_1 (lim4)
########################################################
CUDA_VISIBLE_DEVICES=0 \
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
    python measure_energy_force_uma_gpu_v4.py \
    --dataset_name QH9Stable-random \
    --model_name QHFlow_so2_v5_1_extra_large_260k \
    --uma --num_workers 3 --maxtasksperchild 1000 

########################################################

CUDA_VISIBLE_DEVICES=2 \
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
    python measure_energy_force_uma_gpu_v4.py \
    --dataset_name QH9Stable-size_ood \
    --model_name QHFlow_so2_v5_1_extra_large_260k \
    --uma --num_workers 3 --maxtasksperchild 1000

CUDA_VISIBLE_DEVICES=1 \
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
    python measure_energy_force_uma_gpu_v4.py \
    --dataset_name QH9Dynamic-300k-geometry \
    --model_name QHFlow_so2_v5_1_extra_large_260k \
    --uma --num_workers 3 --maxtasksperchild 1000 --reverse_order

CUDA_VISIBLE_DEVICES=3 \
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
    python measure_energy_force_uma_gpu_v4.py \
    --dataset_name QH9Dynamic-300k-mol \
    --model_name QHFlow_so2_v5_1_extra_large_260k \
    --uma --num_workers 3 --maxtasksperchild 1000 --reverse_order


# large
CUDA_VISIBLE_DEVICES=2 \
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
    python measure_energy_force_uma_gpu_v4.py \
    --dataset_name QH9Stable-random \
    --model_name QHFlow_so2_v5_1_large \
    --uma --num_workers 2 --maxtasksperchild 1000


CUDA_VISIBLE_DEVICES=3 \
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
    python measure_energy_force_uma_gpu_v4.py \
    --dataset_name QH9Stable-size_ood \
    --model_name QHFlow_so2_v5_1_large \
    --uma --num_workers 16 --maxtasksperchild 1000

CUDA_VISIBLE_DEVICES=1 \
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
    python measure_energy_force_uma_gpu_v4.py \
    --dataset_name QH9Dynamic-300k-geometry \
    --model_name QHFlow_so2_v5_1_large \
    --uma --num_workers 16 --maxtasksperchild 1000

CUDA_VISIBLE_DEVICES=0 \
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
    python measure_energy_force_uma_gpu_v4.py \
    --dataset_name QH9Dynamic-300k-mol \
    --model_name QHFlow_so2_v5_1_large \
    --uma --num_workers 16 --maxtasksperchild 1000


## middle
CUDA_VISIBLE_DEVICES=0 \
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
    python measure_energy_force_uma_gpu_v4.py \
    --dataset_name QH9Stable-random \
    --model_name QHFlow_so2_v5_1_middle \
    --uma --num_workers 16 --maxtasksperchild 1000


CUDA_VISIBLE_DEVICES=0 \
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
    python measure_energy_force_uma_gpu_v4.py \
    --dataset_name QH9Stable-size_ood \
    --model_name QHFlow_so2_v5_1_middle \
    --uma --num_workers 16 --maxtasksperchild 1000

CUDA_VISIBLE_DEVICES=0 \
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
    python measure_energy_force_uma_gpu_v4.py \
    --dataset_name QH9Dynamic-300k-geometry \
    --model_name QHFlow_so2_v5_1_middle \
    --uma --num_workers 16 --maxtasksperchild 1000

CUDA_VISIBLE_DEVICES=0 \
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
    python measure_energy_force_uma_gpu_v4.py \
    --dataset_name QH9Dynamic-300k-mol \
    --model_name QHFlow_so2_v5_1_middle \
    --uma --num_workers 16 --maxtasksperchild 1000

## small
CUDA_VISIBLE_DEVICES=1 \
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
    python measure_energy_force_uma_gpu_v4.py \
    --dataset_name QH9Stable-random \
    --model_name QHFlow_so2_v5_1_small \
    --uma --num_workers 16 --maxtasksperchild 1000


CUDA_VISIBLE_DEVICES=0 \
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
    python measure_energy_force_uma_gpu_v4.py \
    --dataset_name QH9Stable-size_ood \
    --model_name QHFlow_so2_v5_1_small \
    --uma --num_workers 16 --maxtasksperchild 1000

CUDA_VISIBLE_DEVICES=0 \
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
    python measure_energy_force_uma_gpu_v4.py \
    --dataset_name QH9Dynamic-300k-geometry \
    --model_name QHFlow_so2_v5_1_small \
    --uma --num_workers 16 --maxtasksperchild 1000

CUDA_VISIBLE_DEVICES=0 \
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
    python measure_energy_force_uma_gpu_v4.py \
    --dataset_name QH9Dynamic-300k-mol \
    --model_name QHFlow_so2_v5_1_small \
    --uma --num_workers 16 --maxtasksperchild 1000


## QHFlow
CUDA_VISIBLE_DEVICES=1 \
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
    python measure_energy_force_uma_gpu_v4.py \
    --dataset_name QH9Stable-random \
    --model_name QHFlow_260k_p1.0 \
    --model_postfix="_IG_True-US_True" \
    --uma --num_workers 3 --maxtasksperchild 1000

CUDA_VISIBLE_DEVICES=0 \
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
    python measure_energy_force_uma_gpu_v4.py \
    --dataset_name QH9Stable-size_ood \
    --model_name QHFlow_260k_p1.0 \
    --model_postfix="_IG_True-US_True" \
    --uma --num_workers 16 --maxtasksperchild 1000

CUDA_VISIBLE_DEVICES=0 \
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
    python measure_energy_force_uma_gpu_v4.py \
    --dataset_name QH9Dynamic-300k-geometry \
    --model_name QHFlow_260k_p1.0 \
    --model_postfix="_IG_True-US_True" \
    --uma --num_workers 16 --maxtasksperchild 1000
    
CUDA_VISIBLE_DEVICES=0 \
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
    python measure_energy_force_uma_gpu_v4.py \
    --dataset_name QH9Dynamic-300k-mol \
    --model_name QHFlow_260k_p1.0 \
    --model_postfix="_IG_True-US_True" \
    --uma --num_workers 16 --maxtasksperchild 1000

# QHNet

CUDA_VISIBLE_DEVICES=1 \
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
    python measure_energy_force_uma_gpu_v4.py \
    --dataset_name QH9Stable-random \
    --model_name Real_QHNet \
    --model_postfix="-UR_False" \
    --uma --num_workers 16 --maxtasksperchild 1000 --no-filtering


CUDA_VISIBLE_DEVICES=0 \
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
    python measure_energy_force_uma_gpu_v4.py \
    --dataset_name QH9Stable-size_ood \
    --model_name Real_QHNet \
    --model_postfix="-UR_False" \
    --uma --num_workers 16 --maxtasksperchild 1000

CUDA_VISIBLE_DEVICES=0 \
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
    python measure_energy_force_uma_gpu_v4.py \
    --dataset_name QH9Dynamic-300k-geometry \
    --model_name Real_QHNet \
    --model_postfix="-UR_False" \
    --uma --num_workers 16 --maxtasksperchild 1000

CUDA_VISIBLE_DEVICES=0 \
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
    python measure_energy_force_uma_gpu_v4.py \
    --dataset_name QH9Dynamic-300k-mol \
    --model_name Real_QHNet \
    --model_postfix="-UR_False" \
    --uma --num_workers 16 --maxtasksperchild 1000

# QHNet_residue

CUDA_VISIBLE_DEVICES=3 \
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
    python measure_energy_force_uma_gpu_v4.py \
    --dataset_name QH9Stable-random \
    --model_name Real_QHNet \
    --model_prefix="_residue" \
    --model_postfix="-UR_True" \
    --uma --num_workers 3 --maxtasksperchild 1000

CUDA_VISIBLE_DEVICES=1 \
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
    python measure_energy_force_uma_gpu_v4.py \
    --dataset_name QH9Stable-random \
    --model_name Real_QHNet \
    --model_prefix="_residue" \
    --model_postfix="-UR_True" \
    --uma --num_workers 3 --tol 1e-8 --pad_eigval None

# Large FT

CUDA_VISIBLE_DEVICES=0 \
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
    python measure_energy_force_uma_gpu_v4.py \
    --dataset_name QH9Stable-random \
    --model_name FT-QHFlow_so2_v5_1_large_lr1e-5_wa10 \
    --model_postfix="-wa" \
    --uma --num_workers 16 --maxtasksperchild 1000

CUDA_VISIBLE_DEVICES=0 \
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
    python measure_energy_force_uma_gpu_v4.py \
    --dataset_name QH9Stable-size_ood \
    --model_name FT-QHFlow_so2_v5_1_large_lr1e-5_wa10 \
    --model_postfix="-wa" \
    --uma --num_workers 16 --maxtasksperchild 1000


CUDA_VISIBLE_DEVICES=0 \
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
    python measure_energy_force_uma_gpu_v4.py \
    --dataset_name QH9Dynamic-300k-mol \
    --model_name FT-QHFlow_so2_v5_1_large_lr1e-5_wa10 \
    --model_postfix="-wa" \
    --uma --num_workers 16 --maxtasksperchild 1000 \
    --start_frac 0.0 --end_frac 0.33

CUDA_VISIBLE_DEVICES=0 \
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
    python measure_energy_force_uma_gpu_v4.py \
    --dataset_name QH9Dynamic-300k-geometry \
    --model_name FT-QHFlow_so2_v5_1_large_lr1e-5_wa10 \
    --model_postfix="-wa" \
    --uma --num_workers 16 --maxtasksperchild 1000 \
    --start_frac 0.0 --end_frac 0.33

# Large FT wav
CUDA_VISIBLE_DEVICES=0 \
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
    python measure_energy_force_uma_gpu_v4.py \
    --dataset_name QH9Stable-random \
    --model_name FT-QHFlow_so2_v5_1_large_lr1e-6_wav2-50 \
    --model_postfix="-wa" \
    --uma --num_workers 16 --maxtasksperchild 1000

CUDA_VISIBLE_DEVICES=0 \
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
    python measure_energy_force_uma_gpu_v4.py \
    --dataset_name QH9Stable-random \
    --model_name FT-QHFlow_so2_v5_1_large_lr1e-6_wav2-100 \
    --model_postfix="-wa" \
    --uma --num_workers 16 --maxtasksperchild 1000

CUDA_VISIBLE_DEVICES=0 \
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
    python measure_energy_force_uma_gpu_v4.py \
    --dataset_name QH9Stable-random \
    --model_name FT-QHFlow_so2_v5_1_large_lr1e-6_wav2-200 \
    --model_postfix="-wa" \
    --uma --num_workers 16 --maxtasksperchild 1000

# single file
CUDA_VISIBLE_DEVICES=0 \
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
    python measure_energy_force_uma_gpu_v4.py \
    --dataset_name QH9Stable-random \
    --model_name FT-QHFlow_so2_v5_1_large_lr1e-6_wav2-50 \
    --model_postfix="-wa" \
    --uma --num_workers 1 --maxtasksperchild 1000 --file_indices "10493"

CUDA_VISIBLE_DEVICES=0 \
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
    python measure_energy_force_uma_gpu_v4.py \
    --dataset_name QH9Stable-random \
    --model_name QHFlow_so2_v5_1_large \
    --uma --num_workers 1 --maxtasksperchild 1000 --file_indices "0,10493"

CUDA_VISIBLE_DEVICES=0 \
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
    python measure_energy_force_uma_gpu_v4.py \
    --dataset_name QH9Stable-random \
    --model_name FT-QHFlow_so2_v5_1_large_lr1e-6_wav4-50 \
    --model_postfix="-wa" \
    --uma --num_workers 1 --maxtasksperchild 1000 --file_indices "10493"


CUDA_VISIBLE_DEVICES=0 \
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
    python measure_energy_force_uma_gpu_v4.py \
    --dataset_name QH9Stable-random \
    --model_name FT-QHFlow_so2_v5_1_large_lr1e-6_wa_fixed2-50 \
    --model_postfix="-wa" \
    --uma --num_workers 1 --maxtasksperchild 1000 --file_indices "10493"

CUDA_VISIBLE_DEVICES=0 \
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
    python measure_energy_force_uma_gpu_v4.py \
    --dataset_name QH9Stable-random \
    --model_name FT-QHFlow_so2_v5_1_small_lr1e-6_wav4_fixed2-50 \
    --model_postfix="-wa" \
    --uma --num_workers 1 --maxtasksperchild 1000 --file_indices "10493"

### WALoss Finetune
CUDA_VISIBLE_DEVICES=1 \
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
    python measure_energy_force_uma_gpu_v4.py \
    --dataset_name QH9Stable-random \
    --model_name FT-QHFlow_so2_v5_1_large_lr1e-6_wa_fixed2-5 \
    --model_postfix="-wa" \
    --uma --num_workers 3 --maxtasksperchild 1000 &&
CUDA_VISIBLE_DEVICES=1 \
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
    python measure_energy_force_uma_gpu_v4.py \
    --dataset_name QH9Stable-random \
    --model_name FT-QHFlow_so2_v5_1_large_lr1e-6_wa_fixed2-20 \
    --model_postfix="-wa" \
    --uma --num_workers 3 --maxtasksperchild 1000 

CUDA_VISIBLE_DEVICES=2 \
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
    python measure_energy_force_uma_gpu_v4.py \
    --dataset_name QH9Stable-random \
    --model_name FT-QHFlow_so2_v5_1_large_lr1e-6_wa_fixed2-50 \
    --model_postfix="-wa" \
    --uma --num_workers 3 --maxtasksperchild 1000 

CUDA_VISIBLE_DEVICES=2 \
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
    python measure_energy_force_uma_gpu_v4.py \
    --dataset_name QH9Stable-random \
    --model_name QHFlow_so2_v5_1_large_ref_hamiltonian \
    --uma --num_workers 3 --maxtasksperchild 1000 

##
CUDA_VISIBLE_DEVICES=3 \
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
    python measure_energy_force_uma_gpu_v4.py \
    --dataset_name QH9Stable-random \
    --model_name QHFlow_so2_v5_1_large \
    --uma --num_workers 2 --maxtasksperchild 1000 &&
CUDA_VISIBLE_DEVICES=3 \
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
    python measure_energy_force_uma_gpu_v4.py \
    --dataset_name QH9Stable-random \
    --model_name QHFlow_so2_v5_1_middle \
    --uma --num_workers 2 --maxtasksperchild 1000 &&
CUDA_VISIBLE_DEVICES=3 \
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
    python measure_energy_force_uma_gpu_v4.py \
    --dataset_name QH9Stable-random \
    --model_name QHFlow_so2_v5_1_small \
    --uma --num_workers 2 --maxtasksperchild 1000

CUDA_VISIBLE_DEVICES=1 \
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
    python measure_energy_force_uma_gpu_v4.py \
    --dataset_name QH9Dynamic-300k-mol \
    --model_name QHFlow_so2_v5_1_large \
    --uma --num_workers 2 --maxtasksperchild 1000 &&
CUDA_VISIBLE_DEVICES=1 \
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
    python measure_energy_force_uma_gpu_v4.py \
    --dataset_name QH9Dynamic-300k-mol \
    --model_name QHFlow_so2_v5_1_middle \
    --uma --num_workers 2 --maxtasksperchild 1000 &&
CUDA_VISIBLE_DEVICES=1 \
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
    python measure_energy_force_uma_gpu_v4.py \
    --dataset_name QH9Dynamic-300k-mol \
    --model_name QHFlow_so2_v5_1_small \
    --uma --num_workers 2 --maxtasksperchild 1000

CUDA_VISIBLE_DEVICES=2 \
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
    python measure_energy_force_uma_gpu_v4.py \
    --dataset_name QH9Stable-size_ood \
    --model_name QHFlow_260k_p1.0 \
    --model_postfix="_IG_True-US_True" \
    --uma --num_workers 3 --maxtasksperchild 1000 &&
CUDA_VISIBLE_DEVICES=2 \
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
    python measure_energy_force_uma_gpu_v4.py \
    --dataset_name QH9Stable-size_ood  \
    --model_name Real_QHNet \
    --model_prefix="_residue" \
    --model_postfix="-UR_True" \
    --uma --num_workers 3 --maxtasksperchild 1000 &&
CUDA_VISIBLE_DEVICES=2 \
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
    python measure_energy_force_uma_gpu_v4.py \
    --dataset_name QH9Stable-size_ood  \
    --model_name Real_QHNet \
    --model_prefix="_residue" \
    --model_postfix="-UR_True" \
    --uma --num_workers 3 --tol 1e-8 --pad_eigval None



########################################################
CUDA_VISIBLE_DEVICES=3 \
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
    python measure_energy_force_uma_gpu_v4.py \
    --dataset_name QH9Dynamic-300k-mol \
    --model_name Real_QHNet \
    --model_postfix="-UR_False" \
    --uma --num_workers 3 --maxtasksperchild 1000 && \
CUDA_VISIBLE_DEVICES=3 \
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
    python measure_energy_force_uma_gpu_v4.py \
    --dataset_name QH9Dynamic-300k-geometry \
    --model_name Real_QHNet \
    --model_postfix="-UR_False" \
    --uma --num_workers 3 --maxtasksperchild 1000 && \
CUDA_VISIBLE_DEVICES=3 \
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
    python measure_energy_force_uma_gpu_v4.py \
    --dataset_name QH9Dynamic-300k-mol \
    --model_name QHFlow_260k_p1.0 \
    --model_postfix="_IG_True-US_True" \
    --uma --num_workers 3 --maxtasksperchild 1000

CUDA_VISIBLE_DEVICES=2 \
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
    python measure_energy_force_uma_gpu_v4.py \
    --dataset_name QH9Stable-size_ood \
    --model_name Real_QHNet \
    --model_postfix="-UR_False" \
    --uma --num_workers 3 --maxtasksperchild 1000