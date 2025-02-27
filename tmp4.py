from models.clip_embedder import CLIPTextEmbedder
import torch
import math
from models.unet_sd_c import UNetModel
from torchinfo import summary

device = torch.device("cuda:1")

model = UNetModel(
    in_channels=1,
    out_channels=1,
    channels=320,
    n_res_blocks=2,
    attention_levels=[1, 2, 3],
    channel_multipliers=[1, 2, 4, 4],
    n_heads=8,
    tf_layers=1,
    d_cond=768,
    # d_cond=None,
).to(device=device)

batch_size = 4
patch_size = 32
num_channel = 1
# summary(
#     model=model,
#     input_size=[
#         (batch_size, num_channel, patch_size, patch_size),
#         (batch_size,),
#         (batch_size, 154, 768),
#     ],
# )

# embedder = CLIPTextEmbedder(device=torch.device("cpu")).eval()
from models.biomedclip_embedder import BiomedCLIPTextEmbedder

embedder = BiomedCLIPTextEmbedder(
    path_json="checkpoints/clip//biomedclip/open_clip_config.json",
    path_bin="checkpoints/clip//biomedclip/open_clip_pytorch_model.bin",
    # context_length=192,
    context_length=256,
    device=device,
).eval()

prompt = ("a b c",) * batch_size
cond = embedder(prompt)
print(cond.shape)
cond = torch.cat([cond, cond], dim=1)
cond = cond.to(device)
# cond = None

# time_steps = torch.ones(size=(batch_size,)).to(device=device)
time_steps = None

x = torch.ones(size=(batch_size, num_channel, patch_size, patch_size)).to(device=device)

with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
    out = model(x, time_steps, cond)
print(out.shape)
