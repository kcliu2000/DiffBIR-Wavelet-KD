import sys
from omegaconf import OmegaConf
from diffbir.utils.common import instantiate_from_config


def count_params(module):
    return sum(p.numel() for p in module.parameters())


cfg_path = sys.argv[1]
cfg = OmegaConf.load(cfg_path)

cldm = instantiate_from_config(cfg.model.cldm)

total = count_params(cldm)
control = count_params(cldm.controlnet)

print("config:", cfg_path)
print(f"total params:      {total/1e6:.2f} M")
print(f"controlnet params: {control/1e6:.2f} M")
