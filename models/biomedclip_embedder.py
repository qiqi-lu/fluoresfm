import json
from open_clip import create_model_and_transforms, get_tokenizer
from open_clip.factory import HF_HUB_PREFIX, _MODEL_CONFIGS
import torch, os
import torch.nn as nn
from typing import List
import numpy as np


class BiomedCLIPTextEmbedder(nn.Module):
    def __init__(
        self,
        path_json="checkpoints/clip//biomedclip/open_clip_config.json",
        path_bin="checkpoints/clip//biomedclip/open_clip_pytorch_model.bin",
        context_length=256,
        device=torch.device("cpu"),
    ):
        super().__init__()

        # Load the model and config files
        model_name = "biomedclip_local"

        with open(path_json, "r") as f:
            config = json.load(f)
            model_cfg = config["model_cfg"]
            preprocess_cfg = config["preprocess_cfg"]

        if (
            not model_name.startswith(HF_HUB_PREFIX)
            and model_name not in _MODEL_CONFIGS
            and config is not None
        ):
            _MODEL_CONFIGS[model_name] = model_cfg

        self.tokenizer = get_tokenizer(model_name)

        self.model, _, preprocess = create_model_and_transforms(
            model_name=model_name,
            pretrained=path_bin,
            **{f"image_{k}": v for k, v in preprocess_cfg.items()},
        )

        self.model.to(device)
        self.device = device
        self.context_length = context_length

        self.textmodel = self.model.text
        self.textmodel.output_tokens = True

    def forward(self, prompts: List[str]):
        texts = self.tokenizer(prompts, context_length=self.context_length).to(
            self.device
        )

        # print(torch.tensor(texts[-1] == 0).sum())

        with torch.no_grad():
            _, text_features = self.textmodel(texts)
        return text_features


if __name__ == "__main__":
    embedder = BiomedCLIPTextEmbedder(context_length=256).eval()
    text = [
        "adenocarcinoma histopathology abd",
        "adenocarcinoma histopathology",
        "task: super-resolution with a scale factor of 2; sample: fixed COS-7 cell line; structure: clathrin-coated pits; wavelength: 488 nm; input microscope: wide-field microscope with excitation numerical aperture of 1.41, detection numerical aperture of 1.3; input pixel size: 62.6 x 62.6 nm, nearest interpolation with a factor of 2; output microscope: linear structured illumination microscope with excitation numerical aperture of 1.41, detection numerical aperture of 1.3; output pixel size: 31.2 x 31.2 nm",
        "task: super-resolution with a scale factor of 2; structure: clathrin-coated pits; input pixel size: 62.6 x 62.6 nm, nearest interpolation with a factor of 2;; output pixel size: 31.2 x 31.2 nm",
        "Task: super-resolution with a scale factor of 2; sample: fixed COS-7 cell line; structure: clathrin-coated pits; fluorescence indicator: mEmerald (GFP); input microscope: wide-field microscope with excitation numerical aperture (NA) of 1.41, detection numerical aperture (NA) of 1.3; input pixel size: 62.6 x 62.6 nm. Nearest interpolation with a factor of 2.; target microscope: linear structured illumination microscope with excitation numerical aperture (NA) of 1.41, detection numerical aperture (NA) of 1.3; target pixel size: 31.3 x 31.3 nm.",
        "Task: super-resolution with a factor of 5 in one dimension; sample: D. melanogaster (Drosophila melanogaster) embryo; structure: histone; fluorescence indicator: mRFP1 (RFP); input microscope: light-sheet microscope with Nikon 10x/0.3 objectives for Illumination, Nikon 16x/0.8 objectives for detection.; input pixel size: 1950 x 390 nm, interpolation with a factor of 5 in the first dimention; target microscope: light-sheet microscope with Nikon 10x/0.3 objectives  for Illumination, Nikon 16x/0.8  objectives for detection.; target pixel size: 390 x 390 nm.",
    ]
    features = embedder(text)
    print(features.shape)
