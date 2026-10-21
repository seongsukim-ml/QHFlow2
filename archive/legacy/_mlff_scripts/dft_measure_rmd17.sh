mamba activate uma
cd ~/25DFT/QHFlow/src/dft_process/post_process

conda activate pyscf-gpu
cd ~/25DFT/QHFlow/src/dft_process/post_process


# huggingface token
huggingface-cli login

CUDA_VISIBLE_DEVICES=0 \
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
    python measure_energy_force_uma_gpu_v5.py \
    --dataset_name rmd-aspirin \
    --model_name QHFlow_so2_v5_1_small_v2_b10 \
    --uma --num_workers 3 --maxtasksperchild 1000 \
    --pos_unit "ang" --gt_xc "pbe" --calc_xc "pbe" --pred_xc "pbe" \
    --pad_eigval 1e-6

CUDA_VISIBLE_DEVICES=0 \
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
    python measure_energy_force_uma_gpu_v5.py \
    --dataset_name rmd-naphthalene \
    --model_name QHFlow_so2_v5_1_small_v2_b10 \
    --uma --num_workers 3 --maxtasksperchild 1000 \
    --pos_unit "ang" --gt_xc "pbe" --calc_xc "pbe" --pred_xc "pbe" \
    --pad_eigval 1e-6

CUDA_VISIBLE_DEVICES=0 \
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
    python measure_energy_force_uma_gpu_v5.py \
    --dataset_name rmd-salicylic_acid \
    --model_name QHFlow_so2_v5_1_small_v2_b10 \
    --uma --num_workers 3 --maxtasksperchild 1000 \
    --pos_unit "ang" --gt_xc "pbe" --calc_xc "pbe" --pred_xc "pbe" \
    --pad_eigval 1e-6

########################################################
CUDA_VISIBLE_DEVICES=0 \
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
    python measure_energy_force_uma_gpu_v5.py \
    --dataset_name rmd-aspirin \
    --model_name QHFlow_so2_v5_1_middle_b10 \
    --uma --num_workers 3 --maxtasksperchild 1000 \
    --pos_unit "ang" --gt_xc "pbe" --calc_xc "pbe" --pred_xc "pbe" \
    --pad_eigval 1e-6

CUDA_VISIBLE_DEVICES=0 \
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
    python measure_energy_force_uma_gpu_v5.py \
    --dataset_name rmd-naphthalene \
    --model_name QHFlow_so2_v5_1_middle_b10 \
    --uma --num_workers 3 --maxtasksperchild 1000 \
    --pos_unit "ang" --gt_xc "pbe" --calc_xc "pbe" --pred_xc "pbe" \
    --pad_eigval 1e-6

CUDA_VISIBLE_DEVICES=0 \
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
    python measure_energy_force_uma_gpu_v5.py \
    --dataset_name rmd-salicylic_acid \
    --model_name QHFlow_so2_v5_1_middle_b10 \
    --uma --num_workers 3 --maxtasksperchild 1000 \
    --pos_unit "ang" --gt_xc "pbe" --calc_xc "pbe" --pred_xc "pbe" \
    --pad_eigval 1e-6