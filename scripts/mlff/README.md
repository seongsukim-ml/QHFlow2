nabla install
```bash
conda create -n nabla python=3.11
conda activate nabla
git clone https://github.com/AIRI-Institute/nablaDFT && cd nablaDFT/
pip install .
```

uma install
```bash
conda create -n uma python=3.12
conda activate uma
pip install fairchem-core
```

uma + qhflow defaults
```bash
conda create -n uma python=3.12  psi4 -y
conda activate uma
pip install fairchem-core==2.6.0
pip install pyscf==2.10.0
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index https://download.pytorch.org/whl/cu121
pip install torch_geometric==2.3.0
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.4.0+cu121.html

pip install -r ~/25DFT/QHFlow/requirements.txt
```

uma + qhflow defaults (cu118 pt 2)
```bash
conda create -n uma2 python=3.12  psi4 -y
conda activate uma2
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index https://download.pytorch.org/whl/cu118
pip install pyscf==2.10.0
pip install torch_geometric==2.3.0
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.4.0+cu118.html

pip install fairchem-core==2.6.0
pip install -r ~/25DFT/QHFlow/requirements.txt
```



uma + qhflow defaults cu124
```bash
conda create -n uma python=3.12  psi4 -y
conda activate uma
pip install fairchem-core==2.6.0
pip install pyscf==2.10.0
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu128
pip install torch_geometric==2.3.0
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.7.0+cu128.html

pip install -r ~/25DFT/QHFlow/requirements.txt
```

pyscf-gpu
```bash
conda create -n pyscf-gpu python=3.12 nvidia/label/cuda-12.4.0::cuda-toolkit cutensor=2.2.0 cupy=13.4.0 psi4 seaborn -y
conda activate pyscf-gpu
pip install nvidia-cuda-runtime-cu12==12.6.37
pip install gpu4pyscf-cuda12x
pip uninstall cupy
conda install cupy=13.4.0
pip install fairchem-core==2.6.0
pip install pyscf==2.10.0
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index https://download.pytorch.org/whl/cu121
pip install torch_geometric==2.3.0
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.4.0+cu121.html
pip install py3Dmol ase
pip install -r ~/25DFT/QHFlow/requirements.txt
```

pyscf-gpu
```bash
mamba create -n pyscf-gpu python=3.12 psi4  -y
conda activate pyscf-gpu
mamba install cuda=12.4.0 -c nvidia
pip install nvidia-cuda-runtime-cu12==12.6.37 seaborn
pip install gpu4pyscf-cuda12x  py3Dmol ase
# pip uninstall cupy
# conda install cupy=13.4.0
pip install fairchem-core==2.6.0
pip install pyscf==2.10.0
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index https://download.pytorch.org/whl/cu121
pip install torch_geometric==2.3.0
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.4.0+cu121.html
pip install -r ~/25DFT/QHFlow/requirements.txt
```

qhflow defaults
```bash
# Example CUDA 12.1 with torch 2.4.1
mamba create -n qhflow python=3.12 psi4 -y
mamba activate qhflow

pip install pyscf==2.10.0
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index https://download.pytorch.org/whl/cu121
pip install torch_geometric==2.3.0
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.4.0+cu121.html

pip install -r ~/25DFT/QHFlow/requirements.txt
```

uma + qhflow defaults
```bash
conda create -n uma python=3.12  psi4 -y
conda activate uma
pip install fairchem-core==2.6.0
pip install pyscf==2.10.0
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install torch_geometric==2.3.0
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.4.0+cu121.html

pip install -r ~/25DFT/QHFlow/requirements.txt
```

uma + qhflow defaults (12.9)
```bash
conda create -n uma129 python=3.12  psi4 -y
conda activate uma129
pip install fairchem-core
pip install pyscf==2.10.0
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu126
pip install torch_geometric
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.6.0+cu126.html

pip install -r ~/25DFT/QHFlow/requirements.txt

```

