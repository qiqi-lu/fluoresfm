"""
A simple test script to check if the model can run.
"""

import torch
from torchinfo import summary
from models.clip_embedder import CLIPTextEmbedder
from models.biomedclip_embedder import BiomedCLIPTextEmbedder
from models.unet_sd_c import UNetModel
from constants import task_struc_micro_voc
from utils.data import IDTokenizer

device = torch.device("cuda:0")

structural_prompt = False
# structural_prompt = True
length_voc = len(task_struc_micro_voc)

print("-" * 80)
print(f"[INFP] Length of token vacabulary: {length_voc}")

# ------------------------------------------------------------------------------
# model
# ------------------------------------------------------------------------------
model = UNetModel(
    in_channels=1,
    out_channels=1,
    channels=320,
    n_res_blocks=1,
    attention_levels=[0, 1, 2, 3],
    channel_multipliers=[1, 2, 4, 4],
    n_heads=8,
    tf_layers=1,
    d_cond=768,
    # d_cond=None,
    structural_prompt=structural_prompt,
    n_tokens=length_voc,
).to(device=device)

# ------------------------------------------------------------------------------
# text embedder
# ------------------------------------------------------------------------------
# embedder = CLIPTextEmbedder(device=torch.device("cpu")).eval()
embedder = BiomedCLIPTextEmbedder(
    path_json="checkpoints/clip//biomedclip/open_clip_config.json",
    path_bin="checkpoints/clip//biomedclip/open_clip_pytorch_model.bin",
    context_length=160,
    device=device,
).eval()

# ------------------------------------------------------------------------------
batch_size = 4
patch_size = 64
num_channel = 1

if not structural_prompt:
    prompt = ("a b c",) * batch_size
    cond = embedder(prompt)
    print(cond.shape)
    cond = torch.cat([cond, cond], dim=1)
    cond = cond.to(device)
    # cond = None
    print(f"[INFO] condition shape: {cond.shape}") if cond is not None else print(cond)
else:
    prompt = (
        "deconvolution;clathrin-coated pits;wide-field microscope;linear structured illumination microscope",
    ) * batch_size
    tokenizer = IDTokenizer(task_struc_micro_voc)
    cond = [tokenizer.encode(p.split(";")) for p in prompt]
    cond = torch.tensor(cond).to(device)
    print(f"[INFO] condition shape: {cond.shape}")

# time_steps = torch.ones(size=(batch_size,)).to(device=device)
time_steps = None

x = torch.ones(size=(batch_size, num_channel, patch_size, patch_size)).to(device=device)

with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
    out = model(x, time_steps, cond)
print(f"[INFO] output shape: {out.shape}")
print("-" * 80)

# ------------------------------------------------------------------------------
from torchinfo import summary

img_lr_shape = (1, patch_size, patch_size)

if not structural_prompt:
    text_empty = (1, 160, 768)
    dtypes = (torch.float16,) * 3
else:
    text_empty = (1, 4)
    dtypes = (torch.float16,) * 2 + (torch.int64,)

with torch.autocast("cuda", torch.float16, enabled=True):

    summary(
        model=model,
        input_size=((1,) + img_lr_shape, (1,), text_empty),
        dtypes=dtypes,
        device=device,
        depth=7,
        col_names=["input_size", "output_size", "num_params", "params_percent"],
    )
