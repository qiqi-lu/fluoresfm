from models.clip_embedder import CLIPTextEmbedder
import torch
from models.unet_sd_c import UNetModel
from torchinfo import summary

device = torch.device("cuda:1")

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
).to(device=device)

batch_size = 4
patch_size = 64
num_channel = 1

# embedder = CLIPTextEmbedder(device=torch.device("cpu")).eval()
from models.biomedclip_embedder import BiomedCLIPTextEmbedder

embedder = BiomedCLIPTextEmbedder(
    path_json="checkpoints/clip//biomedclip/open_clip_config.json",
    path_bin="checkpoints/clip//biomedclip/open_clip_pytorch_model.bin",
    context_length=160,
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

# ------------------------------------------------------------------------------
from torchinfo import summary

img_lr_shape = (1, patch_size, patch_size)
text_shape = (160,)
with torch.autocast("cuda", torch.float16, enabled=True):
    dtype = torch.float16
    summary(
        model=model,
        input_size=((1,) + img_lr_shape, (1,), (1, text_shape[0], 768)),
        dtypes=(dtype,) * 3,
        device=device,
        depth=7,
    )


# ------------------------------------------------------------------------------
# # visualize the model
# import torchviz
# import sys
# import graphviz

# sys.setrecursionlimit(1000000)
# # Generate the dot graph
# dot = torchviz.make_dot(out, params=dict(model.named_parameters()))

# # Save the graph as a PDF
# dot.render("model_visualization", format="pdf")

# ------------------------------------------------------------------------------
# save as onnx file
# with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
#     example_inputs = (x, time_steps, cond)
#     onnx_program = torch.onnx.export(model, example_inputs, dynamo=True)
#     onnx_program.save("model.onnx")
