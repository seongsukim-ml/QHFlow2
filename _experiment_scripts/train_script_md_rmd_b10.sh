conda activate uma && cd ~/25DFT/QHFlow/src

HDF5_USE_FILE_LOCKING=FALSE \
CUDA_VISIBLE_DEVICES=0 python -m experiment.train_md17 \
    dataset=rmd-naphthalene \
    wandb.mode=online \
    mode=predict \
    prefix="_small_v2_b10" \
    model=QHFlow_so2_v5_1_small_v2 \
    dataset.batch_size=10

HDF5_USE_FILE_LOCKING=FALSE \
CUDA_VISIBLE_DEVICES=2 python -m experiment.train_md17 \
    dataset=rmd-salicylic_acid \
    wandb.mode=online \
    prefix="_small_v2_b10" \
    model=QHFlow_so2_v5_1_small_v2 \
    dataset.batch_size=10 \
    mode=predict

HDF5_USE_FILE_LOCKING=FALSE \
CUDA_VISIBLE_DEVICES=2 python -m experiment.train_md17 \
    dataset=rmd-aspirin \
    wandb.mode=online \
    prefix="_small_v2_b10" \
    model=QHFlow_so2_v5_1_small_v2 \
    dataset.batch_size=10 \
    mode=predict

HDF5_USE_FILE_LOCKING=FALSE \
CUDA_VISIBLE_DEVICES=3 python -m experiment.train_md17 \
    dataset=rmd-naphthalene \
    wandb.mode=online \
    prefix="_middle_b10" \
    model=QHFlow_so2_v5_1 \
    dataset.batch_size=10

HDF5_USE_FILE_LOCKING=FALSE \
CUDA_VISIBLE_DEVICES=3 python -m experiment.train_md17 \
    dataset=rmd-salicylic_acid \
    wandb.mode=online \
    prefix="_middle_b10" \
    model=QHFlow_so2_v5_1 \
    dataset.batch_size=10 \
    mode=predict

HDF5_USE_FILE_LOCKING=FALSE \
CUDA_VISIBLE_DEVICES=3 python -m experiment.train_md17 \
    dataset=rmd-aspirin \
    wandb.mode=online \
    prefix="_middle_b10" \
    model=QHFlow_so2_v5_1 \
    dataset.batch_size=10

########################################################

CUDA_VISIBLE_DEVICES=1 python -m experiment.train_md17 \
    dataset=ethanol \
    wandb.mode=online \
    prefix="_small_v2_b10" \
    model=QHFlow_so2_v5_1_small_v2 \
    mode=predict \
    dataset.batch_size=10

CUDA_VISIBLE_DEVICES=6 python -m experiment.train_md17 \
    dataset=malondialdehyde \
    wandb.mode=online \
    prefix="_small_v2_b10" \
    model=QHFlow_so2_v5_1_small_v2 \
    dataset.batch_size=10

CUDA_VISIBLE_DEVICES=3 python -m experiment.train_md17 \
    dataset=uracil \
    wandb.mode=online \
    prefix="_small_v2_b10" \
    model=QHFlow_so2_v5_1_small_v2 \
    dataset.batch_size=10

HDF5_USE_FILE_LOCKING=FALSE \
CUDA_VISIBLE_DEVICES=0 python -m experiment.train_md17 \
    dataset=ethanol \
    wandb.mode=online \
    mode=predict \
    prefix="_middle_b10" \
    model=QHFlow_so2_v5_1 \
    dataset.batch_size=10

HDF5_USE_FILE_LOCKING=FALSE \
CUDA_VISIBLE_DEVICES=1 python -m experiment.train_md17 \
    dataset=malondialdehyde \
    wandb.mode=online \
    prefix="_middle_b10" \
    model=QHFlow_so2_v5_1 \
    dataset.batch_size=10

HDF5_USE_FILE_LOCKING=FALSE \
CUDA_VISIBLE_DEVICES=0 python -m experiment.train_md17 \
    dataset=uracil \
    wandb.mode=online \
    prefix="_middle_b10" \
    model=QHFlow_so2_v5_1 \
    mode=predict \
    dataset.batch_size=10