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
    WanVideoPipeline,
)

# from modelscope import snapshot_download, dataset_snapshot_download
from PIL import Image, ImageFilter

sys.path.append("../../")

# define hight and width
height = 832
width = 480
seed = 0
max_frames = 81
use_teacache = True

test_list_path = [
    # Format: [frame_interval, reference image, driving pose sequence]
    [
        1,
        "data/images/WOMEN-Blouses_Shirts-id_00004955-01_4_full.jpg",
        "data/saved_pose/WOMEN-Blouses_Shirts-id_00004955-01_4_full",
    ],
    # [1, "data/images/musk.jpg", "data/saved_pose/musk"],
    # [
    #     1,
    #     "data/images/WOMEN-Blouses_Shirts-id_00005125-03_4_full.jpg",
    #     "data/saved_pose/WOMEN-Blouses_Shirts-id_00005125-03_4_full",
    # ],
    # [1, "data/images/IMG_20240514_104337.jpg", "data/saved_pose/IMG_20240514_104337"],
    # [1, "data/images/10.jpg", "data/saved_pose/10"],
    # [1, "data/images/taiyi2.jpg", "data/saved_pose/taiyi2"],
]

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

# model_manager.load_lora_v2(
#     "./checkpoints/UniAnimate-Wan2.1-14B-Lora-12000.ckpt", lora_alpha=1.0
# )

model_manager.load_lora_v2(
    "/gen_ca/j1wen/UniAnimate-DiT/models_out/epoch=0-step=1200.ckpt/output_dir/pytorch_model.bin",
    lora_alpha=1.0,
)

# if you use deepspeed to train UniAnimate-Wan2.1, multiple checkpoints may be need to load, use the following form:
# model_manager.load_lora_v2([
#             "./models/lightning_logs/version_0/checkpoints/epoch=2-step=3.ckpt/output_dir/model-00001-of-00011.safetensors",
#             "./models/lightning_logs/version_0/checkpoints/epoch=2-step=3.ckpt/output_dir/model-00002-of-00011.safetensors",
#             "./models/lightning_logs/version_0/checkpoints/epoch=2-step=3.ckpt/output_dir/model-00003-of-00011.safetensors",
#             "./models/lightning_logs/version_0/checkpoints/epoch=2-step=3.ckpt/output_dir/model-00004-of-00011.safetensors",
#             "./models/lightning_logs/version_0/checkpoints/epoch=2-step=3.ckpt/output_dir/model-00005-of-00011.safetensors",
#             "./models/lightning_logs/version_0/checkpoints/epoch=2-step=3.ckpt/output_dir/model-00006-of-00011.safetensors",
#             "./models/lightning_logs/version_0/checkpoints/epoch=2-step=3.ckpt/output_dir/model-00007-of-00011.safetensors",
#             "./models/lightning_logs/version_0/checkpoints/epoch=2-step=3.ckpt/output_dir/model-00008-of-00011.safetensors",
#             "./models/lightning_logs/version_0/checkpoints/epoch=2-step=3.ckpt/output_dir/model-00009-of-00011.safetensors",
#             "./models/lightning_logs/version_0/checkpoints/epoch=2-step=3.ckpt/output_dir/model-00010-of-00011.safetensors",
#             "./models/lightning_logs/version_0/checkpoints/epoch=2-step=3.ckpt/output_dir/model-00011-of-00011.safetensors",
#             ], lora_alpha=1.0)

pipe = WanUniAnimateVideoPipeline.from_model_manager(
    model_manager, torch_dtype=torch.bfloat16, device="cuda"
)
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


for path_dir_per in test_list_path:
    sample_fps = path_dir_per[0]  # frame interval for sampling
    pose_file_path = path_dir_per[2]
    ref_image_path = path_dir_per[1]

    dwpose_all = {}
    frames_all = {}

    for ii_index in sorted(os.listdir(pose_file_path)):
        if ii_index != "pose.jpg":
            dwpose_all[ii_index] = Image.open(os.path.join(pose_file_path, ii_index))
            frames_all[ii_index] = Image.fromarray(
                cv2.cvtColor(cv2.imread(ref_image_path), cv2.COLOR_BGR2RGB)
            )

    frames_pose_ref = Image.open(os.path.join(pose_file_path, "pose.jpg"))

    stride = sample_fps
    _total_frame_num = len(frames_all)
    cover_frame_num = stride * max_frames

    if _total_frame_num < cover_frame_num + 1:
        start_frame = 0
        end_frame = _total_frame_num - 1
        stride = max((_total_frame_num // max_frames), 1)
        end_frame = min(stride * max_frames, _total_frame_num)
    else:
        start_frame = random.randint(0, _total_frame_num - cover_frame_num - 1)
        end_frame = start_frame + cover_frame_num
    frame_list = []
    dwpose_list = []

    random_ref_frame = frames_all[list(frames_all.keys())[0]]
    if random_ref_frame.mode != "RGB":
        random_ref_frame = random_ref_frame.convert("RGB")
    random_ref_dwpose = frames_pose_ref
    if random_ref_dwpose.mode != "RGB":
        random_ref_dwpose = random_ref_dwpose.convert("RGB")

    # sample pose sequence
    for i_index in range(start_frame, end_frame, stride):
        if i_index < len(frames_all):  # Check index within bounds
            i_key = list(frames_all.keys())[i_index]
            i_frame = frames_all[i_key]
            if i_frame.mode != "RGB":
                i_frame = i_frame.convert("RGB")

            i_dwpose = dwpose_all[i_key]
            if i_dwpose.mode != "RGB":
                i_dwpose = i_dwpose.convert("RGB")
            frame_list.append(i_frame)
            dwpose_list.append(i_dwpose)

    if (end_frame - start_frame) < max_frames:
        for _ in range(max_frames - (end_frame - start_frame)):
            i_key = list(frames_all.keys())[end_frame - 1]

            i_frame = frames_all[i_key]
            if i_frame.mode != "RGB":
                i_frame = i_frame.convert("RGB")
            i_dwpose = dwpose_all[i_key]

            frame_list.append(i_frame)
            dwpose_list.append(i_dwpose)

    have_frames = len(frame_list) > 0
    middle_indix = 0

    if have_frames:

        l_hight = random_ref_frame.size[1]
        l_width = random_ref_frame.size[0]

        ref_frame = random_ref_frame

        random_ref_frame_tmp = torchvision.transforms.functional.resize(
            random_ref_frame,
            (height, width),
            interpolation=torchvision.transforms.InterpolationMode.BILINEAR,
        )
        random_ref_dwpose_tmp = resize(random_ref_dwpose)

        dwpose_data_tmp = torch.stack(
            [resize(ss).permute(2, 0, 1) for ss in dwpose_list], dim=0
        )

    dwpose_data = torch.zeros(max_frames, 3, misc_size[0], misc_size[1])

    if have_frames:

        dwpose_data[: len(frame_list), ...] = dwpose_data_tmp

    dwpose_data = dwpose_data.permute(1, 0, 2, 3)

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

    caption = "a person is dancing"
    video_out_condition = []
    for ii in range(dwpose_data_tmp.shape[0]):
        ss = dwpose_list[ii]
        video_out_condition.append(
            image_compose_width(
                random_ref_frame_tmp,
                torchvision.transforms.functional.resize(
                    ss,
                    (height, width),
                    interpolation=torchvision.transforms.InterpolationMode.BILINEAR,
                ),
            )
        )

    # Image-to-video
    video = pipe(
        prompt="a person is dancing",
        negative_prompt="细节模糊不清，字幕，作品，画作，画面，静止，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，杂乱的背景，三条腿，背景人很多，倒着走",
        input_image=ref_frame,
        num_inference_steps=50,
        cfg_scale=1.5,  # slow
        # cfg_scale=1.0, # fast
        seed=seed,
        tiled=True,
        dwpose_data=dwpose_data,
        random_ref_dwpose=random_ref_dwpose_tmp,
        height=height,
        width=width,
        tea_cache_l1_thresh=0.3 if use_teacache else None,
        tea_cache_model_id="Wan2.1-I2V-14B-720P" if use_teacache else None,
    )

    video_out = []
    for ii in range(len(video)):
        ss = video[ii]
        video_out.append(image_compose_width(video_out_condition[ii], ss))
    os.makedirs("./outputs", exist_ok=True)
    save_video(
        video_out,
        "outputs/video_480P_{}_{}.mp4".format(
            ref_image_path.split("/")[-1], pose_file_path.split("/")[-1]
        ),
        fps=15,
        quality=5,
    )

    # CUDA_VISIBLE_DEVICES="0" python examples/unianimate_wan/inference_unianimate_wan_480p.py
