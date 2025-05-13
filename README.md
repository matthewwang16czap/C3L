# C3L

First go to https://github.com/haotian-liu/LLaVA.
Then clone this repository under LLaVA repository.
Then edit pyproject.toml to remove the versions of torch, torchvision, accelerate.

Then remove "metrics" arg at File "/home/matthewwang16czap/projects/LLaVA/llava/train/llava_trainer.py", line 249, in _save_checkpoint
super(LLaVATrainer, self)._save_checkpoint(model, trial, metrics)

Commands for setup conda env:

```bash
conda install -c nvidia cuda-compiler
conda install conda-forge::libstdcxx-ng
pip install fiftyone protobuf==3.20
```

Install LLaVA.

Run scripts
