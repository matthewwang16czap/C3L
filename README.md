# C3L

First go to https://github.com/haotian-liu/LLaVA.
Then clone this repository under LLaVA repository.
Then edit pyproject.toml to remove the versions of torch and torchvision.

Commands for setup conda env:

```bash
conda install -c nvidia cuda-compiler
conda install conda-forge::libstdcxx-ng
pip install fiftyone protobuf
```

Install LLaVA.

Run scripts
