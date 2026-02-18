mamba activate uma
cd ~/25DFT/QHFlow/src/dft_process/post_process

conda activate pyscf-gpu
cd ~/25DFT/QHFlow/src/dft_process/post_process


# huggingface token
huggingface-cli login
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



# CUDA_VISIBLE_DEVICES=0 \
#     OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
#     NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
#     python measure_energy_force_uma_gpu_v5.py \
#     --dataset_name rmd-naphthalene \
#     --model_name Real_QHNet_rmd_v1_lw1 \
#     --model_postfix="-UH_True-UR_True" \
#     --uma --num_workers 3 --maxtasksperchild 1000 \
#     --pos_unit "ang" --gt_xc "pbe" --calc_xc "pbe" --pred_xc "pbe" \
#     --pad_eigval 1e-6 && \
# CUDA_VISIBLE_DEVICES=0 \
#     OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
#     NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
#     python measure_energy_force_uma_gpu_v5.py \
#     --dataset_name rmd-aspirin \
#     --model_name Real_QHNet_rmd_v1_lw1 \
#     --uma --num_workers 3 --maxtasksperchild 1000 \
#     --model_postfix="-UH_True-UR_True" \
#     --pos_unit "ang" --gt_xc "pbe" --calc_xc "pbe" --pred_xc "pbe" \
#     --pad_eigval 1e-6

# CUDA_VISIBLE_DEVICES=3 \
#     OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
#     NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
#     python measure_energy_force_uma_gpu_v5.py \
#     --dataset_name rmd-naphthalene \
#     --model_name QHFlow_so2_v5_1_small_v2_b10 \
#     --uma --num_workers 3 --maxtasksperchild 1000 \
#     --pos_unit "ang" --gt_xc "pbe" --calc_xc "pbe" --pred_xc "pbe" \
#     --pad_eigval 1e-6 && \
# CUDA_VISIBLE_DEVICES=3 \
#     OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
#     NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
#     python measure_energy_force_uma_gpu_v5.py \
#     --dataset_name rmd-salicylic_acid \
#     --model_name Real_QHNet_rmd_v1_lw1 \
#     --model_postfix="-UH_True-UR_True" \
#     --uma --num_workers 3 --maxtasksperchild 1000 \
#     --pos_unit "ang" --gt_xc "pbe" --calc_xc "pbe" --pred_xc "pbe" \
#     --pad_eigval 1e-6


# CUDA_VISIBLE_DEVICES=0 \
#     OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
#     NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
#     python measure_energy_force_uma_gpu_v5.py \
#     --dataset_name uracil \
#     --model_name Real_QHNet \
#     --model_postfix="-UH_True-UR_True" \
#     --uma --num_workers 3 --maxtasksperchild 1000 \
#     --pos_unit "ang" --gt_xc "pbe" --calc_xc "pbe" --pred_xc "pbe" \
#     --pad_eigval 1e-6

# CUDA_VISIBLE_DEVICES=3 \
#     OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
#     NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
#     python measure_energy_force_uma_gpu_v5.py \
#     --dataset_name uracil \
#     --model_name QHFlow_so2_v5_1_small_v2_b10 \
#     --uma --num_workers 3 --maxtasksperchild 1000 \
#     --pos_unit "ang" --gt_xc "pbe" --calc_xc "pbe" --pred_xc "pbe" \
#     --pad_eigval 1e-6

# CUDA_VISIBLE_DEVICES=0 \
#     OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
#     NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
#     python measure_energy_force_uma_gpu_v5.py \
#     --dataset_name uracil \
#     --model_name QHFlow_so2_v5_1_middle_b10_dataset1000 \
#     --uma --num_workers 3 --maxtasksperchild 1000 \
#     --pos_unit "ang" --gt_xc "pbe" --calc_xc "pbe" --pred_xc "pbe" \
#     --pad_eigval 1e-6

# CUDA_VISIBLE_DEVICES=3 \
#     OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
#     NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
#     python measure_energy_force_uma_gpu_v5.py \
#     --dataset_name uracil \
#     --model_name QHFlow_so2_v5_1_middle_b10 \
#     --uma --num_workers 3 --maxtasksperchild 1000 \
#     --pos_unit "ang" --gt_xc "pbe" --calc_xc "pbe" --pred_xc "pbe" \
#     --pad_eigval 1e-6

# CUDA_VISIBLE_DEVICES=1 \
#     OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
#     NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
#     python measure_energy_force_uma_gpu_v5.py \
#     --dataset_name ethanol \
#     --model_name QHFlow_so2_v5_1_small_v2_b10 \
#     --uma --num_workers 3 --maxtasksperchild 1000 \
#     --pos_unit "ang" --gt_xc "pbe" --calc_xc "pbe" --pred_xc "pbe" \
#     --pad_eigval 1e-6

# CUDA_VISIBLE_DEVICES=2 \
#     OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
#     NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
#     python measure_energy_force_uma_gpu_v5.py \
#     --dataset_name rmd-aspirin \
#     --model_name QHFlow_so2_v5_1_small_v2_b10 \
#     --uma --num_workers 3 --maxtasksperchild 1000 \
#     --pos_unit "ang" --gt_xc "pbe" --calc_xc "pbe" --pred_xc "pbe" \
#     --pad_eigval 1e-6

# CUDA_VISIBLE_DEVICES=2 \
#     OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
#     NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
#     python measure_energy_force_uma_gpu_v5.py \
#     --dataset_name rmd-salicylic_acid \
#     --model_name QHFlow_so2_v5_1_middle_b10 \
#     --uma --num_workers 3 --maxtasksperchild 1000 \
#     --pos_unit "ang" --gt_xc "pbe" --calc_xc "pbe" --pred_xc "pbe" \
#     --pad_eigval 1e-6

# CUDA_VISIBLE_DEVICES=3 \
#     OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
#     NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
#     python measure_energy_force_uma_gpu_v5.py \
#     --dataset_name rmd-salicylic_acid \
#     --model_name QHFlow_so2_v5_1_small_v2_b10 \
#     --uma --num_workers 3 --maxtasksperchild 1000 \
#     --pos_unit "ang" --gt_xc "pbe" --calc_xc "pbe" --pred_xc "pbe" \
#     --pad_eigval 1e-6

# CUDA_VISIBLE_DEVICES=0 \
#     OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
#     NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
#     python measure_energy_force_uma_gpu_v5.py \
#     --dataset_name uracil \
#     --model_name Real_QHNet \
#     --model_postfix="-UH_True-UR_False" \
#     --uma --num_workers 3 --maxtasksperchild 1000 \
#     --pos_unit "ang" --gt_xc "pbe" --calc_xc "pbe" --pred_xc "pbe" \
#     --pad_eigval 1e-6 && \
# CUDA_VISIBLE_DEVICES=0 \
#     OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
#     NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
#     python measure_energy_force_uma_gpu_v5.py \
#     --dataset_name ethanol \
#     --model_name Real_QHNet \
#     --model_postfix="-UH_True-UR_False" \
#     --uma --num_workers 3 --maxtasksperchild 1000 \
#     --pos_unit "ang" --gt_xc "pbe" --calc_xc "pbe" --pred_xc "pbe" \
#     --pad_eigval 1e-6 && \
# CUDA_VISIBLE_DEVICES=0 \
#     OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
#     NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
#     python measure_energy_force_uma_gpu_v5.py \
#     --dataset_name malondialdehyde \
#     --model_name Real_QHNet \
#     --model_postfix="-UH_True-UR_False" \
#     --uma --num_workers 3 --maxtasksperchild 1000 \
#     --pos_unit "ang" --gt_xc "pbe" --calc_xc "pbe" --pred_xc "pbe" \
#     --pad_eigval 1e-6


# CUDA_VISIBLE_DEVICES=3 \
#     OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
#     NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
#     python measure_energy_force_uma_gpu_v5.py \
#     --dataset_name rmd-aspirin \
#     --model_name QHFlow_so2_v5_1_middle_b10 \
#     --uma --num_workers 3 --maxtasksperchild 1000 \
#     --pos_unit "ang" --gt_xc "pbe" --calc_xc "pbe" --pred_xc "pbe" \
#     --pad_eigval 1e-6


# CUDA_VISIBLE_DEVICES=3 \
#     OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
#     NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
#     python measure_energy_force_uma_gpu_v5.py \
#     --dataset_name ethanol \
#     --model_name QHFlow_so2_v5_1_middle_b10_UR_False \
#     --uma --num_workers 3 --maxtasksperchild 1000 \
#     --pos_unit "ang" --gt_xc "pbe" --calc_xc "pbe" --pred_xc "pbe" \
#     --pad_eigval 1e-6

# CUDA_VISIBLE_DEVICES=2 \
#     OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
#     NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
#     python measure_energy_force_uma_gpu_v5.py \
#     --dataset_name malondialdehyde \
#     --model_name QHFlow_so2_v5_1_middle_b10_UR_False \
#     --uma --num_workers 3 --maxtasksperchild 1000 \
#     --pos_unit "ang" --gt_xc "pbe" --calc_xc "pbe" --pred_xc "pbe" \
#     --pad_eigval 1e-6

# CUDA_VISIBLE_DEVICES=2 \
#     OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
#     NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
#     python measure_energy_force_uma_gpu_v5.py \
#     --dataset_name uracil \
#     --model_name QHFlow_so2_v5_1_middle_b10_UR_False \
#     --uma --num_workers 3 --maxtasksperchild 1000 \
#     --pos_unit "ang" --gt_xc "pbe" --calc_xc "pbe" --pred_xc "pbe" \
#     --pad_eigval 1e-6

# CUDA_VISIBLE_DEVICES=3 \
#     OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
#     NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
#     python measure_energy_force_uma_gpu_v5.py \
#     --dataset_name malondialdehyde \
#     --model_name QHFlow_so2_v5_1_middle_b10_UR_False \
#     --uma --num_workers 3 --maxtasksperchild 1000 \
#     --pos_unit "ang" --gt_xc "pbe" --calc_xc "pbe" --pred_xc "pbe" \
#     --pad_eigval 1e-6 --no_filtering --pred_tol 1e-6


# CUDA_VISIBLE_DEVICES=2 \
#     OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
#     NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
#     python measure_energy_force_uma_gpu_v5.py \
#     --dataset_name ethanol \
#     --model_name QHFlow_so2_v5_1_middle_b10_ref_init_ham \
#     --uma --num_workers 3 --maxtasksperchild 1000 \
#     --pos_unit "ang" --gt_xc "pbe" --calc_xc "pbe" --pred_xc "pbe" \
#     --pad_eigval 1e-6


# CUDA_VISIBLE_DEVICES=3 \
#     OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
#     NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
#     python measure_energy_force_uma_gpu_v5.py \
#     --dataset_name malondialdehyde \
#     --model_name QHFlow_so2_v5_1_middle_b10_ref_init_ham \
#     --uma --num_workers 3 --maxtasksperchild 1000 \
#     --pos_unit "ang" --gt_xc "pbe" --calc_xc "pbe" --pred_xc "pbe" \
#     --pad_eigval 1e-6

# CUDA_VISIBLE_DEVICES=3 \
#     OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
#     NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
#     python measure_energy_force_uma_gpu_v5.py \
#     --dataset_name rmd-aspirin \
#     --model_name QHFlow_so2_v5_1_middle_b10_train_5k \
#     --uma --num_workers 3 --maxtasksperchild 1000 \
#     --pos_unit "ang" --gt_xc "pbe" --calc_xc "pbe" --pred_xc "pbe" \
#     --pad_eigval 1e-6

# CUDA_VISIBLE_DEVICES=2 \
#     OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
#     NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
#     python measure_energy_force_uma_gpu_v5.py \
#     --dataset_name rmd-aspirin \
#     --model_name QHFlow_so2_v5_1_middle_b10_train_10k \
#     --uma --num_workers 3 --maxtasksperchild 1000 \
#     --pos_unit "ang" --gt_xc "pbe" --calc_xc "pbe" --pred_xc "pbe" \
#     --pad_eigval 1e-6

CUDA_VISIBLE_DEVICES=2 \
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
    python measure_energy_force_uma_gpu_v5.py \
    --dataset_name rmd-aspirin \
    --model_name QHFlow_so2_v5_1_middle_b10_train_15k \
    --uma --num_workers 3 --maxtasksperchild 1000 \
    --pos_unit "ang" --gt_xc "pbe" --calc_xc "pbe" --pred_xc "pbe" \
    --pad_eigval 1e-6

# CUDA_VISIBLE_DEVICES=1 \
#     OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
#     NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
#     python measure_energy_force_uma_gpu_v5.py \
#     --dataset_name rmd-salicylic_acid \
#     --model_name QHFlow_so2_v5_1_middle_b10_train_5k \
#     --uma --num_workers 3 --maxtasksperchild 1000 \
#     --pos_unit "ang" --gt_xc "pbe" --calc_xc "pbe" --pred_xc "pbe" \
#     --pad_eigval 1e-6

# CUDA_VISIBLE_DEVICES=0 \
#     OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
#     NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
#     python measure_energy_force_uma_gpu_v5.py \
#     --dataset_name rmd-salicylic_acid \
#     --model_name QHFlow_so2_v5_1_middle_b10_train_10k \
#     --uma --num_workers 3 --maxtasksperchild 1000 \
#     --pos_unit "ang" --gt_xc "pbe" --calc_xc "pbe" --pred_xc "pbe" \
#     --pad_eigval 1e-6

CUDA_VISIBLE_DEVICES=2 \
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
    python measure_energy_force_uma_gpu_v5.py \
    --dataset_name rmd-salicylic_acid \
    --model_name QHFlow_so2_v5_1_middle_b10_train_15k \
    --uma --num_workers 3 --maxtasksperchild 1000 \
    --pos_unit "ang" --gt_xc "pbe" --calc_xc "pbe" --pred_xc "pbe" \
    --pad_eigval 1e-6

# CUDA_VISIBLE_DEVICES=1 \
#     OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
#     NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
#     python measure_energy_force_uma_gpu_v5.py \
#     --dataset_name rmd-naphthalene \
#     --model_name QHFlow_so2_v5_1_middle_b10_train_5k \
#     --uma --num_workers 3 --maxtasksperchild 1000 \
#     --pos_unit "ang" --gt_xc "pbe" --calc_xc "pbe" --pred_xc "pbe" \
#     --pad_eigval 1e-6

# CUDA_VISIBLE_DEVICES=2 \
#     OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
#     NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
#     python measure_energy_force_uma_gpu_v5.py \
#     --dataset_name rmd-naphthalene \
#     --model_name QHFlow_so2_v5_1_middle_b10_train_10k \
#     --uma --num_workers 3 --maxtasksperchild 1000 \
#     --pos_unit "ang" --gt_xc "pbe" --calc_xc "pbe" --pred_xc "pbe" \
#     --pad_eigval 1e-6

CUDA_VISIBLE_DEVICES=2 \
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
    python measure_energy_force_uma_gpu_v5.py \
    --dataset_name rmd-naphthalene \
    --model_name QHFlow_so2_v5_1_middle_b10_train_15k \
    --uma --num_workers 3 --maxtasksperchild 1000 \
    --pos_unit "ang" --gt_xc "pbe" --calc_xc "pbe" --pred_xc "pbe" \
    --pad_eigval 1e-6

# CUDA_VISIBLE_DEVICES=3 \
#     OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
#     NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
#     python measure_energy_force_uma_gpu_v5.py \
#     --dataset_name malondialdehyde \
#     --model_name QHFlow_so2_v5_1_middle_b20_ref_init_ham \
#     --uma --num_workers 3 --maxtasksperchild 1000 \
#     --pos_unit "ang" --gt_xc "pbe" --calc_xc "pbe" --pred_xc "pbe" \
#     --pad_eigval 1e-6

# CUDA_VISIBLE_DEVICES=2 \
#     OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
#     NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
#     python measure_energy_force_uma_gpu_v5.py \
#     --dataset_name uracil \
#     --model_name QHFlow_so2_v5_1_middle_b20_ref_init_ham \
#     --uma --num_workers 3 --maxtasksperchild 1000 \
#     --pos_unit "ang" --gt_xc "pbe" --calc_xc "pbe" --pred_xc "pbe" \
#     --pad_eigval 1e-6