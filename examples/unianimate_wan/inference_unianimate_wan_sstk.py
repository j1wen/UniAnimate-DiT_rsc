import os
import pickle
import random
import sys
from io import BytesIO

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from diffsynth import (
    ModelManager,
    save_video,
    VideoData,
    WanUniAnimateVideoPipeline,
    WanUniAnimateVideoPipeline_v1,
)

from examples.unianimate_wan.train_unianimate_wan_sstk import (
    shutterstock_video_dataset,
    SSTKVideoDataset_onestage,
)
from PIL import Image, ImageFilter

# define hight and width
height = 832
width = 480
seed = 0
max_frames = 81
use_teacache = True

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str)
parser.add_argument("--save_dir", type=str)

parser.add_argument("--version", type=int, default=0)

# args for ver0
parser.add_argument(
    "--encode_image_dense",
    action="store_true",
    help="Do not add reference poses",
)

# args for ver1
parser.add_argument(
    "--disable_ref_pose",
    action="store_true",
    help="Do not add reference poses",
)
args = parser.parse_args()


def image_compose_width(imag, imag_1):
    # read the size of image1
    rom_image = imag
    width, height = imag.size
    # read the size of image2
    rom_image_1 = imag_1

    width1 = rom_image_1.size[0]
    # create a new image
    to_image = Image.new("RGB", (width + width1, height))
    # paste old images
    to_image.paste(rom_image, (0, 0))
    to_image.paste(rom_image_1, (width, 0))
    return to_image


misc_size = [height, width]

# Download models
# snapshot_download("Wan-AI/Wan2.1-I2V-14B-720P", local_dir="./Wan2.1-I2V-14B-720P")

# Load models
model_manager = ModelManager(device="cpu")
model_manager.load_models(
    ["./Wan2.1-I2V-14B-720P/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"],
    torch_dtype=torch.float32,  # Image Encoder is loaded with float32
)
model_manager.load_models(
    [
        [
            "./Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00001-of-00007.safetensors",
            "./Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00002-of-00007.safetensors",
            "./Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00003-of-00007.safetensors",
            "./Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00004-of-00007.safetensors",
            "./Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00005-of-00007.safetensors",
            "./Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00006-of-00007.safetensors",
            "./Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00007-of-00007.safetensors",
        ],
        "./Wan2.1-I2V-14B-720P/models_t5_umt5-xxl-enc-bf16.pth",
        "./Wan2.1-I2V-14B-720P/Wan2.1_VAE.pth",
    ],
    torch_dtype=torch.bfloat16,  # You can set `torch_dtype=torch.float8_e4m3fn` to enable FP8 quantization.
)

# model_manager.load_lora_v2("./checkpoints/UniAnimate-Wan2.1-14B-Lora-12000.ckpt", lora_alpha=1.0)
model_manager.load_lora_v2(args.model_path, lora_alpha=1.0)

if args.version == 0:
    pipe = WanUniAnimateVideoPipeline.from_model_manager(
        model_manager, torch_dtype=torch.bfloat16, device="cuda"
    )
elif args.version == 1:
    pipe = WanUniAnimateVideoPipeline_v1.from_model_manager(
        model_manager,
        disable_ref_pose=args.disable_ref_pose,
        torch_dtype=torch.bfloat16,
        device="cuda",
    )
if args.encode_image_dense:
    pipe.encode_image = pipe.encode_image_dense
pipe.enable_vram_management(
    num_persistent_param_in_dit=6 * 10**9
)  # You can set `num_persistent_param_in_dit` to a small number to reduce VRAM required.


def resize(image):

    image = torchvision.transforms.functional.resize(
        image,
        (height, width),
        interpolation=torchvision.transforms.InterpolationMode.BILINEAR,
    )
    return torch.from_numpy(np.array(image))


random.seed(seed)
np.random.seed(seed)

shutterstock_video_dataset_v2 = dict(
    # type="LCAShutterstockVideoTrinityDataset",
    data_root="/decoders/suzhaoen/legion/lhm/resampled/full_res_images",
    keypoints_root="/decoders/matthewhu/itw_body_tracking_ls/delivery/sstk_350k/outputs",
    smplx_root="/decoders/junxuanli/legion/lhm/resampled/full_res_images/smplx_params",  # not used
    trinity_root="/decoders/matthewhu/itw_body_tracking_ls/delivery/sstk_350k/outputs",  # used
    mask_root="/decoders/matthewhu/itw_body_tracking_ls/delivery/sstk_350k/outputs",
    index_root="/home/j1wen/rsc/UniAnimate-DiT/data/example_dataset/SSTK350K/val_scenes.txt",
    frame_list_root="/xrcia_shared/ariyanzarei/filtering/SSTK_350K/results/sstk_350k/final_filtered_indices/valid_frames",
    # num_source=num_source,
    # num_target=num_target,
    # repeat_factor=10,  # 1e5 x 10 = 1e6
    # black_background=black_background,
    # face_bbox_aspect_ratio=face_bbox_aspect_ratio,
    # erode_mask=False,
)

dataset = SSTKVideoDataset_onestage(
    **shutterstock_video_dataset_v2,
    max_num_frames=max_frames,
    frame_interval=1,
    num_frames=max_frames,
    height=height,
    width=width,
    is_i2v=True,
    steps_per_epoch=1,
)

selected_indices = np.random.choice(np.arange(len(dataset)), 10, replace=False)
for selected_idx in selected_indices:
    data = dataset[selected_idx]

    first_frame = Image.fromarray(data["first_frame"].numpy())

    video_out_condition = []
    for ii in range(data["dwpose_data"].shape[1]):
        if args.version == 1 and ii == 0:
            ss = Image.fromarray(
                data["random_ref_dwpose_data"].numpy().astype(np.uint8)
            )
        else:
            ss = Image.fromarray(
                data["dwpose_data"][:, ii].permute(1, 2, 0).numpy().astype(np.uint8)
            )
        video_out_condition.append(
            image_compose_width(
                first_frame,
                ss,
            )
        )

    # Image-to-video
    video = pipe(
        prompt="a person is dancing",
        negative_prompt="细节模糊不清，字幕，作品，画作，画面，静止，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，杂乱的背景，三条腿，背景人很多，倒着走",
        input_image=first_frame,
        num_inference_steps=50,
        cfg_scale=1.5,  # slow
        # cfg_scale=1.0, # fast
        seed=seed,
        tiled=True,
        dwpose_data=data["dwpose_data"],
        random_ref_dwpose=data["random_ref_dwpose_data"],
        height=height,
        width=width,
        tea_cache_l1_thresh=0.3 if use_teacache else None,
        tea_cache_model_id="Wan2.1-I2V-14B-720P" if use_teacache else None,
    )

    video_out = []
    for ii in range(len(video)):
        ss = video[ii]
        video_out.append(image_compose_width(video_out_condition[ii], ss))
    os.makedirs(args.save_dir, exist_ok=True)
    save_video(
        video_out,
        "{}/video_480P_{}.mp4".format(args.save_dir, data["path"]),
        fps=15,
        quality=5,
    )

    # CUDA_VISIBLE_DEVICES="0" python examples/unianimate_wan/inference_unianimate_wan_480p.py
