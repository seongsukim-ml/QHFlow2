conda activate uma
cd ~/25DFT/QHFlow/src

## QH9 QHFlow2 small
CUDA_VISIBLE_DEVICES=0 python -m experiment.train_qh9 --config-name config_flow_v2_simple dataset=QH9Stable  dataset.split=random   wandb.mode=online model=QHFlow_so2_v5_1_small  prefix="_small"
CUDA_VISIBLE_DEVICES=1 python -m experiment.train_qh9 --config-name config_flow_v2_simple dataset=QH9Stable  dataset.split=size_ood wandb.mode=online model=QHFlow_so2_v5_1_small  prefix="_small"
CUDA_VISIBLE_DEVICES=2 python -m experiment.train_qh9 --config-name config_flow_v2_simple dataset=QH9Dynamic dataset.split=geometry wandb.mode=online model=QHFlow_so2_v5_1_small  prefix="_small"
CUDA_VISIBLE_DEVICES=3 python -m experiment.train_qh9 --config-name config_flow_v2_simple dataset=QH9Dynamic dataset.split=mol      wandb.mode=online model=QHFlow_so2_v5_1_small  prefix="_small"

## QH9 QHFlow2 middle
CUDA_VISIBLE_DEVICES=0 python -m experiment.train_qh9 --config-name config_flow_v2_simple dataset=QH9Stable  dataset.split=random   wandb.mode=online model=QHFlow_so2_v5_1_middle prefix="_middle"
CUDA_VISIBLE_DEVICES=1 python -m experiment.train_qh9 --config-name config_flow_v2_simple dataset=QH9Stable  dataset.split=size_ood wandb.mode=online model=QHFlow_so2_v5_1_middle prefix="_middle"
CUDA_VISIBLE_DEVICES=2 python -m experiment.train_qh9 --config-name config_flow_v2_simple dataset=QH9Dynamic dataset.split=geometry wandb.mode=online model=QHFlow_so2_v5_1_middle prefix="_middle"
CUDA_VISIBLE_DEVICES=3 python -m experiment.train_qh9 --config-name config_flow_v2_simple dataset=QH9Dynamic dataset.split=mol      wandb.mode=online model=QHFlow_so2_v5_1_middle prefix="_middle"

## QH9 QHFlow2 large
CUDA_VISIBLE_DEVICES=0 python -m experiment.train_qh9 --config-name config_flow_v2_simple dataset=QH9Stable  dataset.split=random   wandb.mode=online model=QHFlow_so2_v5_1_large prefix="_large"
CUDA_VISIBLE_DEVICES=1 python -m experiment.train_qh9 --config-name config_flow_v2_simple dataset=QH9Stable  dataset.split=size_ood wandb.mode=online model=QHFlow_so2_v5_1_large prefix="_large"
CUDA_VISIBLE_DEVICES=2 python -m experiment.train_qh9 --config-name config_flow_v2_simple dataset=QH9Dynamic dataset.split=geometry wandb.mode=online model=QHFlow_so2_v5_1_large prefix="_large"
CUDA_VISIBLE_DEVICES=3 python -m experiment.train_qh9 --config-name config_flow_v2_simple dataset=QH9Dynamic dataset.split=mol      wandb.mode=online model=QHFlow_so2_v5_1_large prefix="_large"

## QH9 QHFlow2 extra large
CUDA_VISIBLE_DEVICES=0 python -m experiment.train_qh9 --config-name config_flow_v2_simple dataset=QH9Stable  dataset.split=random   wandb.mode=online model=QHFlow_so2_v5_1_extra_large prefix="_extra_large"
CUDA_VISIBLE_DEVICES=1 python -m experiment.train_qh9 --config-name config_flow_v2_simple dataset=QH9Stable  dataset.split=size_ood wandb.mode=online model=QHFlow_so2_v5_1_extra_large prefix="_extra_large"
CUDA_VISIBLE_DEVICES=2 python -m experiment.train_qh9 --config-name config_flow_v2_simple dataset=QH9Dynamic dataset.split=geometry wandb.mode=online model=QHFlow_so2_v5_1_extra_large prefix="_extra_large"
CUDA_VISIBLE_DEVICES=3 python -m experiment.train_qh9 --config-name config_flow_v2_simple dataset=QH9Dynamic dataset.split=mol      wandb.mode=online model=QHFlow_so2_v5_1_extra_large prefix="_extra_large"
