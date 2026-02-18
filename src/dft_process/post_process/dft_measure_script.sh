## Uma
conda activate uma
cd ~/25DFT/QHFlow/src/dft_process/post_process

CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 python md17_measure_energy_force_uma.py --dataset_name water --model_name QHFlow --no_parallel
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 python md17_measure_energy_force_uma.py --dataset_name ethanol --model_name QHFlow --no_parallel
CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 python md17_measure_energy_force_uma.py --dataset_name malondialdehyde --model_name QHFlow --no_parallel
CUDA_VISIBLE_DEVICES=3 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 python md17_measure_energy_force_uma.py --dataset_name uracil --model_name QHFlow --no_parallel


CUDA_VISIBLE_DEVICES=0 \
    OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 \
    python measure_energy_force_uma.py \
    --dataset_name QH9Stable-random \
    --model_name Real_QHNet \
    --model_postfix="-UR_False" \
    --no_parallel

CUDA_VISIBLE_DEVICES=1 \
    OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 \
    python measure_energy_force_uma.py \
    --dataset_name QH9Stable-size_ood \
    --model_name Real_QHNet \
    --model_postfix="-UR_False" \
    --no_parallel

CUDA_VISIBLE_DEVICES=2 \
    OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 \
    python measure_energy_force_uma.py \
    --dataset_name QH9Dynamic-300k-geometry \
    --model_name Real_QHNet \
    --model_postfix="-UR_False" \
    --no_parallel

CUDA_VISIBLE_DEVICES=3 \
    OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 \
    python measure_energy_force_uma.py
    --dataset_name QH9Dynamic-300k-mol \
    --model_name Real_QHNet \
    --model_postfix="-UR_False" \
    --no_parallel

CUDA_VISIBLE_DEVICES=0 \
    OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 \
    python measure_energy_force_uma.py \
    --dataset_name QH9Stable-random \
    --model_name Real_QHNet \
    --model_postfix="-UR_False" \
    --no_parallel --uma --start_frac 0.0 --end_frac 0.25

CUDA_VISIBLE_DEVICES=0 \
    OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 \
    python measure_energy_force_uma.py \
    --dataset_name QH9Stable-random \
    --model_name Real_QHNet \
    --model_postfix="-UR_False" \
    --no_parallel --uma --start_frac 0.25 --end_frac 0.5

CUDA_VISIBLE_DEVICES=0 \
    OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 \
    python measure_energy_force_uma.py \
    --dataset_name QH9Stable-random \
    --model_name Real_QHNet \
    --model_postfix="-UR_False" \
    --no_parallel --uma --start_frac 0.5 --end_frac 0.75

CUDA_VISIBLE_DEVICES=0 \
    OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 \
    python measure_energy_force_uma.py \
    --dataset_name QH9Stable-random \
    --model_name Real_QHNet \
    --model_postfix="-UR_False" \
    --no_parallel --uma --start_frac 0.75 --end_frac 1.0