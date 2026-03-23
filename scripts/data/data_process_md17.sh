
python -m dataset_module.qh9_datasets --pdb

# water
conda activate qhflow && cd ~/25DFT/QHFlow/src
python -m dataset_module.md17_datasets_shard --shard_num=2 --shard_idx=0 --prefix="_shard"
python -m dataset_module.md17_datasets_shard --shard_num=2 --shard_idx=1 --prefix="_shard"

# ethanol
conda activate uma && cd ~/25DFT/QHFlow/src
python -m dataset_module.md17_datasets_shard --name=ethanol --shard_num=8 --shard_idx=0-1 --prefix="_shard"
python -m dataset_module.md17_datasets_shard --name=ethanol --shard_num=8 --shard_idx=2-3 --prefix="_shard"
python -m dataset_module.md17_datasets_shard --name=ethanol --shard_num=8 --shard_idx=4-5 --prefix="_shard"
python -m dataset_module.md17_datasets_shard --name=ethanol --shard_num=8 --shard_idx=6-7 --prefix="_shard"

# malondialdehyde
conda activate uma && cd ~/25DFT/QHFlow/src
python -m dataset_module.md17_datasets_shard --name=malondialdehyde --shard_num=8 --shard_idx=0-1 --prefix="_shard"
python -m dataset_module.md17_datasets_shard --name=malondialdehyde --shard_num=8 --shard_idx=2-3 --prefix="_shard"
python -m dataset_module.md17_datasets_shard --name=malondialdehyde --shard_num=8 --shard_idx=4-5 --prefix="_shard"
python -m dataset_module.md17_datasets_shard --name=malondialdehyde --shard_num=8 --shard_idx=6-7 --prefix="_shard"

# uracil
conda activate uma && cd ~/25DFT/QHFlow/src
python -m dataset_module.md17_datasets_shard --name=uracil --shard_num=16 --shard_idx=0-1 --prefix="_shard"
python -m dataset_module.md17_datasets_shard --name=uracil --shard_num=16 --shard_idx=2-3 --prefix="_shard"
python -m dataset_module.md17_datasets_shard --name=uracil --shard_num=16 --shard_idx=4-5 --prefix="_shard"
python -m dataset_module.md17_datasets_shard --name=uracil --shard_num=16 --shard_idx=6-7 --prefix="_shard"
python -m dataset_module.md17_datasets_shard --name=uracil --shard_num=16 --shard_idx=8-9 --prefix="_shard"
python -m dataset_module.md17_datasets_shard --name=uracil --shard_num=16 --shard_idx=10-11 --prefix="_shard"
python -m dataset_module.md17_datasets_shard --name=uracil --shard_num=16 --shard_idx=12-13 --prefix="_shard"
python -m dataset_module.md17_datasets_shard --name=uracil --shard_num=16 --shard_idx=14-15 --prefix="_shard"

conda activate qhflow && cd ~/25DFT/QHFlow/src
python -m dataset_module.md17_datasets_shard --name=uracil --shard_num=16 --shard_idx=0 --prefix="_shard"
python -m dataset_module.md17_datasets_shard --name=uracil --shard_num=16 --shard_idx=1 --prefix="_shard"
python -m dataset_module.md17_datasets_shard --name=uracil --shard_num=16 --shard_idx=2 --prefix="_shard"
python -m dataset_module.md17_datasets_shard --name=uracil --shard_num=16 --shard_idx=3 --prefix="_shard"
python -m dataset_module.md17_datasets_shard --name=uracil --shard_num=16 --shard_idx=4 --prefix="_shard"
python -m dataset_module.md17_datasets_shard --name=uracil --shard_num=16 --shard_idx=5 --prefix="_shard"
python -m dataset_module.md17_datasets_shard --name=uracil --shard_num=16 --shard_idx=6 --prefix="_shard"
python -m dataset_module.md17_datasets_shard --name=uracil --shard_num=16 --shard_idx=7 --prefix="_shard"

python -m dataset_module.md17_datasets_shard --name=uracil --shard_num=16 --shard_idx=8 --prefix="_shard"
python -m dataset_module.md17_datasets_shard --name=uracil --shard_num=16 --shard_idx=9 --prefix="_shard"
python -m dataset_module.md17_datasets_shard --name=uracil --shard_num=16 --shard_idx=10 --prefix="_shard"
python -m dataset_module.md17_datasets_shard --name=uracil --shard_num=16 --shard_idx=11 --prefix="_shard"
python -m dataset_module.md17_datasets_shard --name=uracil --shard_num=16 --shard_idx=12 --prefix="_shard"
python -m dataset_module.md17_datasets_shard --name=uracil --shard_num=16 --shard_idx=13 --prefix="_shard"
python -m dataset_module.md17_datasets_shard --name=uracil --shard_num=16 --shard_idx=14 --prefix="_shard"
python -m dataset_module.md17_datasets_shard --name=uracil --shard_num=16 --shard_idx=15 --prefix="_shard"
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

# svr6
python -m dataset_module.qh9_datasets_shard --shard_num=60 --shard_idx=30-34 --prefix="_shard"
python -m dataset_module.qh9_datasets_shard --shard_num=60 --shard_idx=35-39 --prefix="_shard"
python -m dataset_module.qh9_datasets_shard --shard_num=60 --shard_idx=40-44 --prefix="_shard"

# svr10
python -m dataset_module.qh9_datasets_shard --shard_num=60 --shard_idx=45-49 --prefix="_shard"
python -m dataset_module.qh9_datasets_shard --shard_num=60 --shard_idx=50-54 --prefix="_shard"
python -m dataset_module.qh9_datasets_shard --shard_num=60 --shard_idx=55-59 --prefix="_shard"

# Test
python -m dataset_module.qh9_datasets_shard --shard_num=60 --prefix="_shard" --use_ref_hamiltonian_as_init_ham
python -m dataset_module.qh9_datasets_shard --shard_num=60 --prefix="_shard" --name=QH9Dynamic --use_ref_hamiltonian_as_init_ham
