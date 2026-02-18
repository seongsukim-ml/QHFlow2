
python -m dataset_module.qh9_datasets --pdb

# sample
conda activate qhflow && cd ~/25DFT/QHFlow/src
python -m dataset_module.qh9_datasets_shard --shard_num=10 --shard_idx=0-1 --prefix="_sample"
python -m dataset_module.qh9_datasets_shard --shard_num=10 --shard_idx=2-3 --prefix="_sample"
python -m dataset_module.qh9_datasets_shard --shard_num=10 --shard_idx=4-5 --prefix="_sample"
python -m dataset_module.qh9_datasets_shard --shard_num=10 --shard_idx=6-7 --prefix="_sample"
python -m dataset_module.qh9_datasets_shard --shard_num=10 --shard_idx=8-9 --prefix="_sample"


# sample
conda activate qhflow && cd ~/25DFT/QHFlow/src
python -m dataset_module.qh9_datasets_shard --shard_num=10 --shard_idx=0-1 --name=QH9Dynamic
python -m dataset_module.qh9_datasets_shard --shard_num=10 --shard_idx=2-3 --name=QH9Dynamic
python -m dataset_module.qh9_datasets_shard --shard_num=10 --shard_idx=4-5 --name=QH9Dynamic
python -m dataset_module.qh9_datasets_shard --shard_num=10 --shard_idx=6-7 --name=QH9Dynamic
python -m dataset_module.qh9_datasets_shard --shard_num=10 --shard_idx=8-9 --name=QH9Dynamic

# shard
# svr11
python -m dataset_module.qh9_datasets_shard --shard_num=60 --shard_idx=0-4 --prefix="_shard"   
python -m dataset_module.qh9_datasets_shard --shard_num=60 --shard_idx=5-9 --prefix="_shard"
python -m dataset_module.qh9_datasets_shard --shard_num=60 --shard_idx=10-14 --prefix="_shard"
python -m dataset_module.qh9_datasets_shard --shard_num=60 --shard_idx=15-19 --prefix="_shard"
python -m dataset_module.qh9_datasets_shard --shard_num=60 --shard_idx=20-24 --prefix="_shard"
python -m dataset_module.qh9_datasets_shard --shard_num=60 --shard_idx=25-29 --prefix="_shard"
python -m dataset_module.qh9_datasets_shard --shard_num=60 --shard_idx=30-34 --prefix="_shard"
python -m dataset_module.qh9_datasets_shard --shard_num=60 --shard_idx=35-39 --prefix="_shard"
python -m dataset_module.qh9_datasets_shard --shard_num=60 --shard_idx=40-44 --prefix="_shard"

# svr10
python -m dataset_module.qh9_datasets_shard --shard_num=60 --shard_idx=45-49 --prefix="_shard"
python -m dataset_module.qh9_datasets_shard --shard_num=60 --shard_idx=50-54 --prefix="_shard"
python -m dataset_module.qh9_datasets_shard --shard_num=60 --shard_idx=55-59 --prefix="_shard"

python -m dataset_module.qh9_datasets_shard --shard_num=60 --shard_idx=55 --prefix="_shard"
python -m dataset_module.qh9_datasets_shard --shard_num=60 --shard_idx=56 --prefix="_shard"
python -m dataset_module.qh9_datasets_shard --shard_num=60 --shard_idx=57 --prefix="_shard"
python -m dataset_module.qh9_datasets_shard --shard_num=60 --shard_idx=58 --prefix="_shard"
python -m dataset_module.qh9_datasets_shard --shard_num=60 --shard_idx=59 --prefix="_shard"


conda activate qhflow && cd ~/25DFT/QHFlow/src
conda activate uma    && cd ~/25DFT/QHFlow/src

python -m dataset_module.qh9_datasets_shard --shard_num=30 --shard_idx=-1 --name=QH9Dynamic --prefix="_shard"

# svr6
python -m dataset_module.qh9_datasets_shard --shard_num=60 --shard_idx=0-4 --name=QH9Dynamic   --prefix="_shard"
python -m dataset_module.qh9_datasets_shard --shard_num=60 --shard_idx=5-9 --name=QH9Dynamic   --prefix="_shard"
python -m dataset_module.qh9_datasets_shard --shard_num=60 --shard_idx=10-14 --name=QH9Dynamic --prefix="_shard"
python -m dataset_module.qh9_datasets_shard --shard_num=60 --shard_idx=15-19 --name=QH9Dynamic --prefix="_shard"

# saturn 2
python -m dataset_module.qh9_datasets_shard --shard_num=60 --shard_idx=20-24 --name=QH9Dynamic --prefix="_shard"
python -m dataset_module.qh9_datasets_shard --shard_num=60 --shard_idx=25-29 --name=QH9Dynamic --prefix="_shard"
python -m dataset_module.qh9_datasets_shard --shard_num=60 --shard_idx=30-34 --name=QH9Dynamic --prefix="_shard"
python -m dataset_module.qh9_datasets_shard --shard_num=60 --shard_idx=35-39 --name=QH9Dynamic --prefix="_shard"

# saturn 1
python -m dataset_module.qh9_datasets_shard --shard_num=60 --shard_idx=40-44 --name=QH9Dynamic --prefix="_shard"
python -m dataset_module.qh9_datasets_shard --shard_num=60 --shard_idx=45-49 --name=QH9Dynamic --prefix="_shard"
python -m dataset_module.qh9_datasets_shard --shard_num=60 --shard_idx=50-54 --name=QH9Dynamic --prefix="_shard"
python -m dataset_module.qh9_datasets_shard --shard_num=60 --shard_idx=55-59 --name=QH9Dynamic --prefix="_shard"


# svr11hw
python -m dataset_module.qh9_datasets_shard --shard_num=60 --shard_idx=24 --name=QH9Dynamic --prefix="_shard"
python -m dataset_module.qh9_datasets_shard --shard_num=60 --shard_idx=29 --name=QH9Dynamic --prefix="_shard"
python -m dataset_module.qh9_datasets_shard --shard_num=60 --shard_idx=34 --name=QH9Dynamic --prefix="_shard"
python -m dataset_module.qh9_datasets_shard --shard_num=60 --shard_idx=39 --name=QH9Dynamic --prefix="_shard"

# svr10hw
python -m dataset_module.qh9_datasets_shard --shard_num=60 --shard_idx=44 --name=QH9Dynamic --prefix="_shard"
python -m dataset_module.qh9_datasets_shard --shard_num=60 --shard_idx=49 --name=QH9Dynamic --prefix="_shard"
python -m dataset_module.qh9_datasets_shard --shard_num=60 --shard_idx=54 --name=QH9Dynamic --prefix="_shard"
python -m dataset_module.qh9_datasets_shard --shard_num=60 --shard_idx=59 --name=QH9Dynamic --prefix="_shard"

python -m dataset_module.qh9_datasets_shard --shard_num=60 --shard_idx=9 --name=QH9Dynamic --prefix="_shard"
python -m dataset_module.qh9_datasets_shard --shard_num=60 --shard_idx=10 --name=QH9Dynamic --prefix="_shard"

# Rowan
conda activate qhflow && cd ~/25DFT/QHFlow/src

python -m dataset_module.qh9_datasets_rowan --shard_num=90 --shard_idx=0-9 --name=QM9Rowan
python -m dataset_module.qh9_datasets_rowan --shard_num=90 --shard_idx=9-18 --name=QM9Rowan
python -m dataset_module.qh9_datasets_rowan --shard_num=90 --shard_idx=18-27 --name=QM9Rowan
python -m dataset_module.qh9_datasets_rowan --shard_num=90 --shard_idx=27-36 --name=QM9Rowan
python -m dataset_module.qh9_datasets_rowan --shard_num=90 --shard_idx=36-45 --name=QM9Rowan

python -m dataset_module.qh9_datasets_rowan --shard_num=90 --shard_idx=45-54 --name=QM9Rowan
python -m dataset_module.qh9_datasets_rowan --shard_num=90 --shard_idx=54-63 --name=QM9Rowan
python -m dataset_module.qh9_datasets_rowan --shard_num=90 --shard_idx=63-72 --name=QM9Rowan
python -m dataset_module.qh9_datasets_rowan --shard_num=90 --shard_idx=72-81 --name=QM9Rowan
python -m dataset_module.qh9_datasets_rowan --shard_num=90 --shard_idx=81-89 --name=QM9Rowan

## Testing code
python -m dataset_module.qh9_datasets_shard --shard_num=60 --shard_idx=0 --name=QH9Stable --prefix="_shard" --pdb
