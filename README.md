# C3L

First go to https://github.com/haotian-liu/LLaVA.
Then clone this repository under LLaVA repository.
Then edit `pyproject.toml` to remove the versions of `torch`, `torchvision`, `accelerate`.

Then remove `metrics` arg at File "/home/username/projects/LLaVA/llava/train/llava_trainer.py", line 249, in \_save_checkpoint
`super(LLaVATrainer, self)._save_checkpoint(model, trial, metrics)`

Then add `cache_position=None` to the `forward()` method in Class `LlavaLlamaForCausalLM`.

Commands for setup conda env:

```bash
conda install -c nvidia cuda-compiler
conda install conda-forge::libstdcxx-ng
pip install fiftyone openpyxl protobuf==3.20
```

Install LLaVA.

Run scripts
