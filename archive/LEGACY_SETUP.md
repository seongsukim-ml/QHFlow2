# Legacy installation examples

These Torch 2.4 / CUDA 12.1 examples were moved from the main README for
historical reference. They do not provide native Muon and are not the runtime
used for the updated v0.2.0 models. Use the
[Stable/Muon runtime](../docs/QH9STABLE_MUON.md#stable-data-and-runtime) and
[requirements-stable-muon.txt](../requirements-stable-muon.txt) for those models.

Run the historical commands below from the repository root. They are retained
as previous environment recipes, not a newly validated installation.

```bash
# Example CUDA 12.1 with torch 2.4.1 without gpu-pyscf
conda create -n qhflow2 python=3.12 -y
conda activate qhflow2

pip install fairchem-core==2.6.0
pip install pyscf==2.10.0
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index https://download.pytorch.org/whl/cu121
pip install torch_geometric==2.3.0
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.4.0+cu121.html

pip install -r requirements.txt
```

```bash
# Example CUDA 12.1 with torch 2.4.1 with gpu-pyscf
conda create -n qhflow2-gpu-pyscf python=3.12 -y
conda activate qhflow2-gpu-pyscf
conda install cuda=12.4.0 -c nvidia

pip install nvidia-cuda-runtime-cu12==12.6.37
pip install gpu4pyscf-cuda12x
pip uninstall cupy
conda install cupy=13.4.0

pip install fairchem-core==2.6.0
pip install pyscf==2.10.0
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index https://download.pytorch.org/whl/cu121
pip install torch_geometric==2.3.0
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.4.0+cu121.html

pip install -r requirements.txt
```

See [archived usage](LEGACY_USAGE.md) for historical training commands.
