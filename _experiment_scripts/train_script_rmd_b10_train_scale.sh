conda activate pyscf-gpu2 && cd ~/25DFT/QHFlow/src

# escflow small (svr6hw)
# HDF5_USE_FILE_LOCKING=FALSE \
lim 3
HDF5_USE_FILE_LOCKING=FALSE \
CUDA_VISIBLE_DEVICES=0 python -m experiment.train_md17 \
    dataset=rmd-naphthalene \
    wandb.mode=online \
    prefix="_middle_b10_train_10k" \
    model=QHFlow_so2_v5_1 \
    dataset.batch_size=10 \
    +dataset.train_size=10000

HDF5_USE_FILE_LOCKING=FALSE \
CUDA_VISIBLE_DEVICES=1 python -m experiment.train_md17 \
    dataset=rmd-salicylic_acid \
    wandb.mode=online \
    prefix="_middle_b10_train_10k" \
    model=QHFlow_so2_v5_1 \
    dataset.batch_size=10 \
    +dataset.train_size=10000

HDF5_USE_FILE_LOCKING=FALSE \
CUDA_VISIBLE_DEVICES=0 python -m experiment.train_md17 \
    dataset=rmd-aspirin \
    wandb.mode=online \
    prefix="_middle_b10_train_10k" \
    model=QHFlow_so2_v5_1 \
    dataset.batch_size=10 \
    +dataset.train_size=10000

########################################################

HDF5_USE_FILE_LOCKING=FALSE \
CUDA_VISIBLE_DEVICES=2 python -m experiment.train_md17 \
    dataset=rmd-naphthalene \
    wandb.mode=online \
    prefix="_middle_b10_train_5k" \
    model=QHFlow_so2_v5_1 \
    dataset.batch_size=10 \
    +dataset.train_size=5000

HDF5_USE_FILE_LOCKING=FALSE \
CUDA_VISIBLE_DEVICES=1 python -m experiment.train_md17 \
    dataset=rmd-salicylic_acid \
    wandb.mode=online \
    prefix="_middle_b10_train_5k" \
    model=QHFlow_so2_v5_1 \
    dataset.batch_size=10 \
    +dataset.train_size=5000

HDF5_USE_FILE_LOCKING=FALSE \
CUDA_VISIBLE_DEVICES=0 python -m experiment.train_md17 \
    dataset=rmd-aspirin \
    wandb.mode=online \
    prefix="_middle_b10_train_5k" \
    model=QHFlow_so2_v5_1 \
    dataset.batch_size=10 \
    +dataset.train_size=5000

########################################################

HDF5_USE_FILE_LOCKING=FALSE \
CUDA_VISIBLE_DEVICES=2 python -m experiment.train_md17 \
    dataset=rmd-naphthalene \
    wandb.mode=online \
    prefix="_middle_b10_train_2_5k" \
    model=QHFlow_so2_v5_1 \
    dataset.batch_size=10 \
    +dataset.train_size=2500

HDF5_USE_FILE_LOCKING=FALSE \
CUDA_VISIBLE_DEVICES=3 python -m experiment.train_md17 \
    dataset=rmd-salicylic_acid \
    wandb.mode=online \
    prefix="_middle_b10_train_2_5k" \
    model=QHFlow_so2_v5_1 \
    dataset.batch_size=10 \
    +dataset.train_size=2500

HDF5_USE_FILE_LOCKING=FALSE \
CUDA_VISIBLE_DEVICES=3 python -m experiment.train_md17 \
    dataset=rmd-aspirin \
    wandb.mode=online \
    prefix="_middle_b10_train_2_5k" \
    model=QHFlow_so2_v5_1 \
    dataset.batch_size=10 \
    +dataset.train_size=2500

########################################################

HDF5_USE_FILE_LOCKING=FALSE \
CUDA_VISIBLE_DEVICES=2 python -m experiment.train_md17 \
    dataset=rmd-naphthalene \
    wandb.mode=online \
    prefix="_middle_b10_train_1k" \
    model=QHFlow_so2_v5_1 \
    dataset.batch_size=10 \
    +dataset.train_size=1000

HDF5_USE_FILE_LOCKING=FALSE \
CUDA_VISIBLE_DEVICES=3 python -m experiment.train_md17 \
    dataset=rmd-salicylic_acid \
    wandb.mode=online \
    prefix="_middle_b10_train_1k" \
    model=QHFlow_so2_v5_1 \
    dataset.batch_size=10 \
    +dataset.train_size=1000

HDF5_USE_FILE_LOCKING=FALSE \
CUDA_VISIBLE_DEVICES=3 python -m experiment.train_md17 \
    dataset=rmd-aspirin \
    wandb.mode=online \
    prefix="_middle_b10_train_1k" \
    model=QHFlow_so2_v5_1 \
    dataset.batch_size=10 \
    +dataset.train_size=1000