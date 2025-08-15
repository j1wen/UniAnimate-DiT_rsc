import argparse, imageio, os, torch
import contextlib
import copy

import datetime
import glob
import gzip
import json
import pickle
import random
import sys
import time
from io import BytesIO

import cv2
import numpy as np
import pandas as pd

import pytorch_lightning as pl

import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from diffsynth import (
    load_state_dict,
    load_state_dict_from_folder,
    ModelManager,
    WanVideoPipeline,
)
from einops import rearrange
from peft import inject_adapter_in_model, LoraConfig
from PIL import Image, ImageFilter

from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.plugins.environments import LightningEnvironment
from pytorch_lightning.strategies import SingleDeviceStrategy
from torchvision.transforms import v2
from tqdm import tqdm

from train_util import coco_wholebody2openpose, draw_keypoints

os.environ["TOKENIZERS_PARALLELISM"] = "false"

TB_FERQ = 100


@contextlib.contextmanager
def suppress_stderr():
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    stderr_fd = sys.stderr.fileno()
    sys.stderr.flush()
    saved_stderr_fd = os.dup(stderr_fd)

    os.dup2(devnull_fd, stderr_fd)
    os.close(devnull_fd)

    try:
        yield
    finally:
        os.dup2(saved_stderr_fd, stderr_fd)
        os.close(saved_stderr_fd)


shutterstock_video_dataset = dict(
    # type="LCAShutterstockVideoTrinityDataset",
    data_root="/decoders/suzhaoen/legion/lhm/resampled/full_res_images",
    keypoints_root="/decoders/junxuanli/legion/lhm/resampled/full_res_images/keypoints",
    smplx_root="/decoders/junxuanli/legion/lhm/resampled/full_res_images/smplx_params",
    trinity_root="/decoders/marcopesavento/datasets_new/LCA_trinity/trinity_poses",
    mask_root="/gen_ca/data/legion/lhm/resampled/derived/alpha_masks_vit",
    index_root="/decoders/junxuanli/legion/lhm/resampled/full_res_images/filter_joint_v5/index",
    frame_list_root="/decoders/marcopesavento/datasets_new/LCA_trinity/valid_files_corrected_new",
    # num_source=num_source,
    # num_target=num_target,
    # repeat_factor=10,  # 1e5 x 10 = 1e6
    # black_background=black_background,
    # face_bbox_aspect_ratio=face_bbox_aspect_ratio,
    # erode_mask=False,
)


shutterstock_video_dataset_v2 = dict(
    # type="LCAShutterstockVideoTrinityDataset",
    data_root="/decoders/suzhaoen/legion/lhm/resampled/full_res_images",
    keypoints_root="/decoders/junxuanli/legion/lhm/resampled/full_res_images/keypoints",
    smplx_root="/decoders/junxuanli/legion/lhm/resampled/full_res_images/smplx_params",
    trinity_root="/decoders/marcopesavento/datasets_new/LCA_trinity/trinity_poses",
    mask_root="/gen_ca/data/legion/lhm/resampled/derived/alpha_masks_vit",
    index_root="/decoders/junxuanli/legion/lhm/resampled/full_res_images/filter_joint_v5/index",
    frame_list_root="/decoders/marcopesavento/datasets_new/LCA_trinity/valid_files_corrected_new",
    # num_source=num_source,
    # num_target=num_target,
    # repeat_factor=10,  # 1e5 x 10 = 1e6
    # black_background=black_background,
    # face_bbox_aspect_ratio=face_bbox_aspect_ratio,
    # erode_mask=False,
)


class SSTKVideoDataset_onestage(torch.utils.data.Dataset):
    def __init__(
        self,
        data_root,
        keypoints_root,
        smplx_root,
        trinity_root,
        mask_root,
        index_root,
        frame_list_root,
        max_num_frames=81,
        frame_interval=2,
        num_frames=81,
        height=480,
        width=832,
        is_i2v=False,
        steps_per_epoch=1,
        load_face=False,
    ):
        # metadata = pd.read_csv(metadata_path)
        # self.path = [os.path.join(base_path, "train", file_name) for file_name in metadata["file_name"]]
        # self.text = metadata["text"].to_list()
        self.data_root = data_root
        self.mask_root = mask_root
        self.smplx_root = smplx_root
        self.trinity_root = trinity_root
        self.keypoints_root = keypoints_root
        self.index_root = index_root
        self.frame_list_root = frame_list_root
        self.num_source = 1
        self.num_target = num_frames
        self.load_face = load_face

        self.data_list = self.load_data_list()

        self.max_num_frames = max_num_frames
        self.frame_interval = frame_interval
        self.num_frames = num_frames
        self.height = height
        self.width = width
        self.is_i2v = is_i2v
        self.steps_per_epoch = steps_per_epoch

        # data_list = ['UBC_Fashion', 'self_collected_videos_pose', 'TikTok']
        data_list = ["TikTok"]
        self.sample_fps = frame_interval
        self.max_frames = max_num_frames
        self.misc_size = [height, width]
        self.video_list = []

        # if "TikTok" in data_list:

        #     self.pose_dir = "./data/example_dataset/TikTok/"
        #     file_list = os.listdir(self.pose_dir)
        #     print("!!! all dataset length: ", len(file_list))
        #     #
        #     for iii_index in file_list:
        #         self.video_list.append(self.pose_dir + iii_index)

        #     self.use_pose = True
        #     print("!!! dataset length: ", len(self.video_list))

        # if "UBC_Fashion" in data_list:
        #     self.pose_dir = "path_of_UBC_Fashion"
        #     file_list = os.listdir(self.pose_dir)
        #     print("!!! all dataset length (UBC_Fashion): ", len(file_list))

        #     for iii_index in file_list:
        #         #
        #         self.video_list.append(self.pose_dir + iii_index)

        #     self.use_pose = True
        #     print("!!! dataset length: ", len(self.video_list))
        # if "self_collected_videos_pose" in data_list:

        #     self.pose_dir = "path_of_your_self_data"
        #     file_list = os.listdir(self.pose_dir)
        #     print(
        #         "!!! all dataset length (self_collected_videos_pose): ", len(file_list)
        #     )
        #     #
        #     for iii_index in file_list:

        #         self.video_list.append(self.pose_dir + iii_index)

        #     self.use_pose = True
        #     print("!!! dataset length: ", len(self.video_list))

        random.shuffle(self.video_list)

        self.frame_process = v2.Compose(
            [
                # v2.CenterCrop(size=(height, width)),
                v2.Resize(size=(height, width), antialias=True),
                v2.ToTensor(),
                v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

    def load_data_list(self):
        data_list = []

        self.rgb_dir = self.data_root
        self.mask_dir = (
            os.path.join(self.data_root, "mask")
            if self.mask_root is None
            else self.mask_root
        )
        self.smplx_dir = (
            os.path.join(self.data_root, "smplx")
            if self.smplx_root is None
            else self.smplx_root
        )
        self.trinity_dir = self.trinity_root
        self.pose_dir = (
            os.path.join(self.data_root, "keypoints")
            if self.keypoints_root is None
            else self.keypoints_root
        )
        self.index_dir = self.index_root
        self.frame_list_dir = self.frame_list_root

        video_names = []

        if self.index_dir.endswith(".txt") or self.index_dir.endswith(".csv"):
            with open(self.index_dir, "r") as f:
                lines = f.readlines()
            for line in lines:
                line = line.strip()
                if len(line) == 0:
                    continue
                if line.startswith("uuid,"):
                    continue
                line = line.split(",")[0]
                video_names.append(line)
        else:
            index_files = sorted(
                glob.glob(os.path.join(self.index_dir, "partition_*.csv"))
            )
            for index_file in tqdm(index_files):
                with open(index_file, "r") as f:
                    lines = f.readlines()
                for line in lines:
                    if line.startswith("uuid,"):
                        continue
                    line = line.strip()
                    if len(line) == 0:
                        continue
                    line = line.split(",")
                    video_name = line[0]
                    video_names.append(video_name)

        self.video_names_valid = video_names

        with open("invalid_video_names.txt") as f:
            invalid_paths = [line.split(" ")[0] for line in f.readlines()[:-1]]
        filtered_video_names = []
        for video_name in self.video_names_valid:
            if video_name not in invalid_paths:
                filtered_video_names.append(video_name)
        self.video_names_valid = filtered_video_names

        return data_list

    def resize(self, image):
        width, height = image.size
        #
        image = torchvision.transforms.functional.resize(
            image,
            (self.height, self.width),
            interpolation=torchvision.transforms.InterpolationMode.BILINEAR,
        )
        # return torch.from_numpy(np.array(image))
        return image

    def crop_and_resize(self, image):
        width, height = image.size
        scale = max(self.width / width, self.height / height)
        image = torchvision.transforms.functional.resize(
            image,
            (round(height * scale), round(width * scale)),
            interpolation=torchvision.transforms.InterpolationMode.BILINEAR,
        )
        return image

    def name2parts(self, cid):
        if len(cid) > 4:
            parts = [cid[i : i + 4] for i in range(0, len(cid), 4)]
            cid_parts = "/".join(parts)
        else:
            cid_parts = cid
        return cid_parts

    def process_video(self, args):
        (
            video_name,
            rgb_dir,
            mask_dir,
            smplx_dir,
            trinity_dir,
            pose_dir,
            frame_list_dir,
            num_target,
        ) = args

        video_name_parts = self.name2parts(video_name)

        # Define subdirectories for the video
        image_dir = os.path.join(rgb_dir, video_name_parts)
        mask_video_dir = os.path.join(mask_dir, video_name_parts)
        smplx_video_dir = os.path.join(smplx_dir, video_name_parts, "smplx_params")
        if not os.path.exists(smplx_video_dir):
            smplx_video_dir = os.path.join(smplx_dir, video_name_parts)
        trinity_video_dir = os.path.join(trinity_dir, video_name_parts, "smplx_params")
        if not os.path.exists(trinity_video_dir):
            trinity_video_dir = os.path.join(trinity_dir, video_name_parts)
        pose_video_dir = os.path.join(pose_dir, video_name_parts)
        frame_list_file = os.path.join(
            frame_list_dir, video_name_parts, f"{video_name}_valid_name_list.txt"
        )
        if not os.path.exists(frame_list_file):
            frame_list_file = os.path.join(
                frame_list_dir,
                video_name_parts,
                "smplx_params",
                f"{video_name}_valid_name_list.txt",
            )

        # Check if all required directories exist
        if not (
            os.path.exists(image_dir)
            and os.path.exists(mask_video_dir)
            # and os.path.exists(smplx_video_dir)
            and os.path.exists(pose_video_dir)
            and os.path.exists(frame_list_file)
        ):
            missing_path = []
            if not os.path.exists(image_dir):
                missing_path.append(image_dir)
            if not os.path.exists(mask_video_dir):
                missing_path.append(mask_video_dir)
            # if not os.path.exists(smplx_video_dir):
            #     missing_path.append(smplx_video_dir)
            if not os.path.exists(pose_video_dir):
                missing_path.append(pose_video_dir)
            if not os.path.exists(frame_list_file):
                missing_path.append(frame_list_file)
            return video_name, [], 0, "missing path: " + ",".join(missing_path)

        # Collect file names (without extensions)
        with open(frame_list_file, "r") as f:
            all_images_file = f.readlines()
            common_names = [x.strip() for x in all_images_file]

        rgb_extension = os.listdir(image_dir)[0].split(".")[-1]
        assert rgb_extension == "jpg" or rgb_extension == "png"

        # Build the list of dictionaries for this video
        video_data_list = []
        for name in common_names:
            data_info = {
                "rgb_path": os.path.join(image_dir, name + "." + rgb_extension),
                "mask_path": os.path.join(mask_video_dir, name + ".png"),
                "smplx_path": os.path.join(smplx_video_dir, name + ".json"),
                # "trinity_path": os.path.join(trinity_video_dir, name + ".txt"),
                "pose_path": os.path.join(pose_video_dir, name + ".json"),
            }
            missing_assets = False
            for k, v in data_info.items():
                if not os.path.exists(v):
                    missing_assets = True
                    continue
            if missing_assets:
                continue
            data_info["video_name"] = video_name
            video_data_list.append(data_info)

        # Ensure there are enough samples in this video
        if len(video_data_list) < num_target:
            # print(f"not enough frame in {video_name}")
            return video_name, [], 0, "not enough frame: " + str(len(video_data_list))

        return video_name, video_data_list, len(video_data_list), "success"

    def get_data_info(self, idx):
        idx = idx % len(self.video_names_valid)  ## repeat_factor
        source_video_name = self.video_names_valid[idx]

        args = (
            source_video_name,
            self.rgb_dir,
            self.mask_dir,
            self.smplx_dir,
            self.trinity_dir,
            self.pose_dir,
            self.frame_list_dir,
            self.num_target,
        )
        _, video_data_list, num_valid_data_list, _ = self.process_video(args)
        if num_valid_data_list == 0:
            return None

        ## for shape consistency, grab the beta from the first frame
        video_idxs = [i for i in range(len(video_data_list))]
        first_idx = video_idxs[0]
        first_data_info = video_data_list[first_idx]
        video_shape_params = json.load(open(first_data_info["smplx_path"]))[
            "betas"
        ]  ## 10

        source_idxs = random.sample(video_idxs, self.num_source)
        if self.load_face:
            source_idxs_face = random.sample(video_idxs, self.num_source)
        source_data_list = []
        for i in range(len(source_idxs)):
            source_idx = source_idxs[i]
            source_data_info = copy.deepcopy(video_data_list[source_idx])
            source_data = self.get_data_info_helper(
                source_data_info, video_shape_params=video_shape_params, with_face=True
            )

            if self.load_face:
                source_idx_face = source_idxs_face[i]
                source_data_info_face = copy.deepcopy(video_data_list[source_idx_face])
                source_data_face = self.get_data_info_helper(
                    source_data_info_face,
                    video_shape_params=video_shape_params,
                    with_face=True,
                )

            if source_data is None or (self.load_face and source_data_face is None):
                return None

            if self.load_face:
                face_expr_image = source_data["face_img"]
                source_data["face_img"] = source_data_face["face_img"]
                source_data["face_expr_image"] = face_expr_image

            source_data_list.append(source_data)

        ## randomly sample num_target target images
        # random sample fps
        stride = random.randint(1, self.sample_fps)

        _total_frame_num = len(video_idxs)
        cover_frame_num = stride * self.max_frames
        max_frames = self.max_frames
        if _total_frame_num < cover_frame_num + 1:
            start_frame = 0
            end_frame = _total_frame_num - 1
            stride = max((_total_frame_num // max_frames), 1)
            end_frame = min(stride * max_frames, _total_frame_num - 1)
        else:
            start_frame = random.randint(0, _total_frame_num - cover_frame_num)
            end_frame = start_frame + cover_frame_num

        target_data_list = []
        for target_idx in range(start_frame, end_frame, stride):
            target_data_info = copy.deepcopy(video_data_list[target_idx])
            target_data = self.get_data_info_helper(
                target_data_info, video_shape_params=video_shape_params, with_face=True
            )

            assert target_data_info["video_name"] == source_video_name

            if target_data is None:
                return None

            target_data_list.append(target_data)

        data = {"source_list": source_data_list, "target_list": target_data_list}
        return data

    def get_data_info_helper(
        self, data_info, video_shape_params=None, with_face=False, validate_mask=False
    ):
        with suppress_stderr():
            rgb_path = data_info["rgb_path"]
            img = cv2.imread(data_info["rgb_path"])[
                :, :, [2, 1, 0]
            ]  ## bgr image is default
            mask = cv2.imread(data_info["mask_path"])  # don't need mask

        # matte = mask[:, :, 0].astype(np.float32) / 255.0

        # if self.erode_mask:
        #     kernel = np.ones((3, 3), np.uint8)
        #     mask = cv2.erode(mask, kernel, iterations=1)

        mask = (mask[:, :, 0] > 128).astype(np.uint8) * 255

        # if self.black_background:
        #     cond_img = (img * matte[:, :, None] + 255 * (1 - matte[:, :, None])).astype(
        #         np.uint8
        #     )
        # else:
        cond_img = img

        # body_pose_params = json.load(open(data_info["smplx_path"]))

        ## coco 133 wholebody keypoints, pixel aligned
        pose = json.load(open(data_info["pose_path"]))
        assert len(pose["instance_info"]) > 0
        keypoints = pose["instance_info"][0]["keypoints"]
        keypoint_scores = pose["instance_info"][0]["keypoint_scores"]

        keypoints = (
            np.array(keypoints).reshape(-1, 2).astype(np.float32)
        )  ## K x 2; x, y; K = 133
        keypoint_scores = np.array(keypoint_scores)  ## K

        keypoints_openpose = coco_wholebody2openpose(keypoints)

        H, W = img.shape[:2]
        canvas_without_face, canvas = draw_keypoints(keypoints_openpose, (H, W))

        video_name = data_info["video_name"]

        if img is None:  # or mask is None:  # or body_pose_params is None:
            return None

        # ## too few valid pixels
        # if (mask > 0).sum() < 8:
        #     return None

        # if "expr" not in body_pose_params:
        #     body_pose_params["expr"] = np.zeros(100)  ## 100

        # image_height, image_width = img.shape[:2]
        # fallback_indices = np.array([0, 1, 2, 3, 4])
        # valid_points = keypoints[fallback_indices]
        # face_bbox = self.get_face_bbox(valid_points, image_height, image_width)
        # if with_face == True:
        #     face_img = self.crop_image(cond_img, bbox=face_bbox)
        # else:
        #     face_img = None

        # if video_shape_params is not None:
        #     body_pose_params["betas"] = (
        #         video_shape_params  ## swap with beta from first frame
        #     )

        # K = self.get_intrinsics(body_pose_params=body_pose_params)  ## 3 x 3
        # M = np.eye(4)  ## 4 x 4

        # ##----------------------------------------
        # ## pad image and mask
        # target_width, target_height = body_pose_params[
        #     "img_size_wh"
        # ]  ## this is 1.2 image size
        # img, mask, face_bbox, matte, cond_img = self.pad_image_mask(
        #     img,
        #     mask,
        #     target_height,
        #     target_width,
        #     face_bbox,
        #     matte=matte,
        #     cond_img=cond_img,
        # )
        # assert img.shape[:2] == (target_height, target_width)
        # assert mask.shape[:2] == (target_height, target_width)

        # # trinity pose
        # with open(data_info["trinity_path"], "r") as f:
        #     lines = f.readlines()
        # trinity_pose = np.array([float(x) for x in lines[:204]], dtype=np.float32)
        # trinity_params = {
        #     "lbs_motion": trinity_pose[:138],
        #     "lbs_scale": trinity_pose[138:204],
        # }

        data_info = {
            "img": img,
            "body_img": Image.fromarray(cond_img),
            # "face_img": face_img,
            # "face_bbox": face_bbox,
            "img_id": os.path.basename(data_info["rgb_path"]),
            "img_path": data_info["rgb_path"],
            "pose_img": Image.fromarray(canvas_without_face),
            "mask": mask,  # shape: H x W (uint8, with 255 being fg mask, 0 being bg mask, 128 being padded mask)
            # "matte": matte,  # shape: H x W (float32, with 1 being fg, 0 being bg)
            # "body_pose_params": trinity_params,
            # "K": K,
            # "M": M,
            "video_name": video_name,
        }
        # data_info["body_img"].save("temp/{}".format(rgb_path.split("/")[-1]))
        # Image.fromarray(
        #     (
        #         np.array(data_info["pose_img"]) * 0.5
        #         + np.array(data_info["body_img"]) * 0.5
        #     ).astype(np.uint8)
        # ).save("temp/{}".format("pose_" + rgb_path.split("/")[-1]))

        return data_info

    # def __getitem__(self, data_id):
    def __getitem__(self, index):
        index = index % len(self.video_names_valid)
        data = self.get_data_info(index)

        success = False
        frame_list = []
        dwpose_list = []
        mask_list = []
        try:
            clean = True

            random_ref_frame = data["source_list"][0]["body_img"]
            if random_ref_frame.mode != "RGB":
                random_ref_frame = random_ref_frame.convert("RGB")
            random_ref_dwpose = data["source_list"][0]["pose_img"]
            random_ref_mask = data["source_list"][0]["mask"]

            first_frame = None
            for i_index in range(len(data["target_list"])):
                i_frame = data["target_list"][i_index]["body_img"]
                if i_frame.mode != "RGB":
                    i_frame = i_frame.convert("RGB")
                i_dwpose = data["target_list"][i_index]["pose_img"]
                i_mask = data["target_list"][i_index]["mask"]

                if first_frame is None:
                    first_frame = i_frame

                    frame_list.append(i_frame)
                    dwpose_list.append(i_dwpose)
                    mask_list.append(i_mask)

                else:
                    frame_list.append(i_frame)
                    dwpose_list.append(i_dwpose)
                    mask_list.append(i_mask)

            max_frames = self.max_frames
            if len(data["target_list"]) < max_frames:
                for _ in range(max_frames - len(data["target_list"])):
                    i_key = len(data["target_list"]) - 1

                    i_frame = data["target_list"][i_key]["body_img"]
                    if i_frame.mode != "RGB":
                        i_frame = i_frame.convert("RGB")
                    i_dwpose = data["target_list"][i_key]["pose_img"]
                    i_mask = data["target_list"][i_key]["mask"]

                    frame_list.append(i_frame)
                    dwpose_list.append(i_dwpose)
                    mask_list.append(i_mask)

            have_frames = len(data["target_list"]) > 0
            middle_indix = 0

            if have_frames:

                l_hight = random_ref_frame.size[1]
                l_width = random_ref_frame.size[0]

                # center crop the reference image to a vertical image
                target_crop_width = int(self.height / l_hight * l_width)
                random_center_y, random_center_x = np.median(
                    np.stack(random_ref_mask.nonzero()), axis=1
                )
                random_ref_frame = random_ref_frame.crop(
                    (
                        random_center_x - target_crop_width // 2,
                        0,
                        random_center_x + target_crop_width // 2,
                        l_hight,
                    )
                )
                random_ref_dwpose = random_ref_dwpose.crop(
                    (
                        random_center_x - target_crop_width // 2,
                        0,
                        random_center_x + target_crop_width // 2,
                        l_hight,
                    )
                )

                # now center crop the generated video sequence using first frame's mask
                center_y, center_x = np.median(np.stack(mask_list[0].nonzero()), axis=1)
                dwpose_list_new, frame_list_new = [], []
                for dwpose, frame in zip(dwpose_list, frame_list):
                    dwpose = dwpose.crop(
                        (
                            center_x - target_crop_width // 2,
                            0,
                            center_x + target_crop_width // 2,
                            l_hight,
                        )
                    )
                    frame = frame.crop(
                        (
                            center_x - target_crop_width // 2,
                            0,
                            center_x + target_crop_width // 2,
                            l_hight,
                        )
                    )
                    dwpose_list_new.append(dwpose)
                    frame_list_new.append(frame)

                dwpose_list = dwpose_list_new
                frame_list = frame_list_new

                # replace with new height and width
                l_hight = random_ref_frame.size[1]
                l_width = random_ref_frame.size[0]

                # random crop
                x1 = random.randint(0, l_width // 14)
                x2 = random.randint(0, l_width // 14)
                y1 = random.randint(0, l_hight // 14)
                y2 = random.randint(0, l_hight // 14)

                random_ref_frame = random_ref_frame.crop(
                    (x1, y1, l_width - x2, l_hight - y2)
                )
                ref_frame = random_ref_frame
                #

                random_ref_frame_tmp = torch.from_numpy(
                    np.array(self.resize(random_ref_frame))
                )
                random_ref_dwpose_tmp = torch.from_numpy(
                    np.array(
                        self.resize(
                            random_ref_dwpose.crop((x1, y1, l_width - x2, l_hight - y2))
                        )
                    )
                )  # [3, 512, 320]

                video_data_tmp = torch.stack(
                    [self.frame_process(self.resize(random_ref_frame))]
                    + [
                        self.frame_process(
                            self.resize(ss.crop((x1, y1, l_width - x2, l_hight - y2)))
                        )
                        for ss in frame_list
                    ],
                    dim=0,
                )  # self.transforms(frames)
                dwpose_data_tmp = torch.stack(
                    [
                        torch.from_numpy(
                            np.array(
                                self.resize(
                                    random_ref_dwpose.crop(
                                        (x1, y1, l_width - x2, l_hight - y2)
                                    )
                                )
                            )
                        ).permute(2, 0, 1)
                    ]
                    + [
                        torch.from_numpy(
                            np.array(
                                self.resize(ss.crop((x1, y1, l_width - x2, l_hight - y2)))
                            )
                        ).permute(2, 0, 1)
                        for ss in dwpose_list
                    ],
                    dim=0,
                )

                # Image.fromarray(random_ref_dwpose_tmp.numpy().astype(np.uint8)).save(
                #     "temp/ref_pose.png"
                # )
                # Image.fromarray(random_ref_frame_tmp.numpy().astype(np.uint8)).save(
                #     "temp/ref.png"
                # )
                # for i, (frame, dwpose) in enumerate(zip(video_data_tmp, dwpose_data_tmp)):
                #     Image.fromarray(
                #         ((frame + 0.5) * 0.5 * 255)
                #         .numpy()
                #         .astype(np.uint8)
                #         .transpose(1, 2, 0)
                #     ).save(f"temp/{i:06d}.png")
                #     Image.fromarray(
                #         dwpose.numpy().astype(np.uint8).transpose(1, 2, 0)
                #     ).save(f"temp/{i:06d}_pose.png")

            video_data = torch.zeros(
                self.max_frames + 1, 3, self.misc_size[0], self.misc_size[1]
            )
            dwpose_data = torch.zeros(
                self.max_frames + 1, 3, self.misc_size[0], self.misc_size[1]
            )

            if have_frames:
                video_data[: len(frame_list) + 1, ...] = video_data_tmp

                dwpose_data[: len(frame_list) + 1, ...] = dwpose_data_tmp

            video_data = video_data.permute(1, 0, 2, 3)
            dwpose_data = dwpose_data.permute(1, 0, 2, 3)

            caption = "a person is dancing"
        except Exception as e:
            #
            caption = "a person is dancing"
            #
            video_data = torch.zeros(
                3, self.max_frames + 1, self.misc_size[0], self.misc_size[1]
            )
            random_ref_frame_tmp = torch.zeros(
                self.misc_size[0], self.misc_size[1], 3
            ).int()
            vit_image = torch.zeros(3, self.misc_size[0], self.misc_size[1])

            dwpose_data = torch.zeros(
                3, self.max_frames + 1, self.misc_size[0], self.misc_size[1]
            )
            #
            random_ref_dwpose_tmp = torch.zeros(self.misc_size[0], self.misc_size[1], 3)
            print(
                "{} read video frame failed with error: {}".format(
                    "".join(self.video_names_valid[index]), e
                )
            )

        text = caption
        path = "".join(self.video_names_valid[index])

        if self.is_i2v:
            video, first_frame = video_data, random_ref_frame_tmp
            data = {
                "text": text,
                "video": video,
                "path": path,
                "first_frame": first_frame,
                "dwpose_data": dwpose_data,
                "random_ref_dwpose_data": random_ref_dwpose_tmp,
            }
        else:
            data = {"text": text, "video": video, "path": path}
        return data

    def __len__(self):

        return len(self.video_names_valid)


class LightningModelForTrain_onestage(pl.LightningModule):
    def __init__(
        self,
        dit_path,
        learning_rate=1e-5,
        lora_rank=4,
        lora_alpha=4,
        train_architecture="lora",
        lora_target_modules="q,k,v,o,ffn.0,ffn.2",
        init_lora_weights="kaiming",
        use_gradient_checkpointing=True,
        use_gradient_checkpointing_offload=False,
        pretrained_lora_path=None,
        model_VAE=None,
        #
    ):
        super().__init__()
        model_manager = ModelManager(torch_dtype=torch.bfloat16, device="cpu")
        if os.path.isfile(dit_path):
            model_manager.load_models([dit_path])
        else:
            dit_path = dit_path.split(",")
            model_manager.load_models([dit_path])

        self.pipe = WanVideoPipeline.from_model_manager(model_manager)
        self.pipe.scheduler.set_timesteps(1000, training=True)

        self.pipe_VAE = model_VAE.pipe.eval()
        self.tiler_kwargs = model_VAE.tiler_kwargs

        concat_dim = 4
        self.dwpose_embedding = nn.Sequential(
            nn.Conv3d(
                3, concat_dim * 4, (3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)
            ),
            nn.SiLU(),
            nn.Conv3d(
                concat_dim * 4,
                concat_dim * 4,
                (3, 3, 3),
                stride=(1, 1, 1),
                padding=(1, 1, 1),
            ),
            nn.SiLU(),
            nn.Conv3d(
                concat_dim * 4,
                concat_dim * 4,
                (3, 3, 3),
                stride=(1, 1, 1),
                padding=(1, 1, 1),
            ),
            nn.SiLU(),
            nn.Conv3d(
                concat_dim * 4,
                concat_dim * 4,
                (3, 3, 3),
                stride=(1, 2, 2),
                padding=(1, 1, 1),
            ),
            nn.SiLU(),
            nn.Conv3d(concat_dim * 4, concat_dim * 4, 3, stride=(2, 2, 2), padding=1),
            nn.SiLU(),
            nn.Conv3d(concat_dim * 4, concat_dim * 4, 3, stride=(2, 2, 2), padding=1),
            nn.SiLU(),
            nn.Conv3d(concat_dim * 4, 5120, (1, 2, 2), stride=(1, 2, 2), padding=0),
        )

        randomref_dim = 20
        self.randomref_embedding_pose = nn.Sequential(
            nn.Conv2d(3, concat_dim * 4, 3, stride=1, padding=1),
            nn.SiLU(),
            nn.Conv2d(concat_dim * 4, concat_dim * 4, 3, stride=1, padding=1),
            nn.SiLU(),
            nn.Conv2d(concat_dim * 4, concat_dim * 4, 3, stride=1, padding=1),
            nn.SiLU(),
            nn.Conv2d(concat_dim * 4, concat_dim * 4, 3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(concat_dim * 4, concat_dim * 4, 3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(concat_dim * 4, randomref_dim, 3, stride=2, padding=1),
        )
        self.freeze_parameters()

        # self.freeze_parameters()
        if train_architecture == "lora":
            self.add_lora_to_model(
                self.pipe.denoising_model(),
                lora_rank=lora_rank,
                lora_alpha=lora_alpha,
                lora_target_modules=lora_target_modules,
                init_lora_weights=init_lora_weights,
                pretrained_lora_path=pretrained_lora_path,
            )
        else:
            self.pipe.denoising_model().requires_grad_(True)

        self.learning_rate = learning_rate
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_gradient_checkpointing_offload = use_gradient_checkpointing_offload

    def freeze_parameters(self):
        # Freeze parameters
        self.pipe.requires_grad_(False)
        self.pipe.eval()
        self.pipe.denoising_model().train()
        self.pipe_VAE.requires_grad_(False)
        self.pipe_VAE.eval()
        self.randomref_embedding_pose.train()
        self.dwpose_embedding.train()

    def add_lora_to_model(
        self,
        model,
        lora_rank=4,
        lora_alpha=4,
        lora_target_modules="q,k,v,o,ffn.0,ffn.2",
        init_lora_weights="kaiming",
        pretrained_lora_path=None,
        state_dict_converter=None,
    ):
        # Add LoRA to UNet
        self.lora_alpha = lora_alpha
        if init_lora_weights == "kaiming":
            init_lora_weights = True

        lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            init_lora_weights=init_lora_weights,
            target_modules=lora_target_modules.split(","),
        )
        model = inject_adapter_in_model(lora_config, model)
        for param in model.parameters():
            # Upcast LoRA parameters into fp32
            if param.requires_grad:
                param.data = param.to(torch.float32)

        # Lora pretrained lora weights
        if pretrained_lora_path is not None:
            #
            try:
                state_dict = load_state_dict(pretrained_lora_path)
            except:
                state_dict = load_state_dict_from_folder(pretrained_lora_path)
            #
            state_dict_new = {}
            state_dict_new_module = {}
            for key in state_dict.keys():

                if "pipe.dit." in key:
                    key_new = key.split("pipe.dit.")[1]
                    state_dict_new[key_new] = state_dict[key]
                if "dwpose_embedding" in key or "randomref_embedding_pose" in key:
                    state_dict_new_module[key] = state_dict[key]
            state_dict = state_dict_new
            state_dict_new = {}

            for key in state_dict_new_module:
                if "dwpose_embedding" in key:
                    state_dict_new[key.split("dwpose_embedding.")[1]] = (
                        state_dict_new_module[key]
                    )
            self.dwpose_embedding.load_state_dict(state_dict_new, strict=True)

            state_dict_new = {}
            for key in state_dict_new_module:
                if "randomref_embedding_pose" in key:
                    state_dict_new[key.split("randomref_embedding_pose.")[1]] = (
                        state_dict_new_module[key]
                    )
            self.randomref_embedding_pose.load_state_dict(state_dict_new, strict=True)

            if state_dict_converter is not None:
                state_dict = state_dict_converter(state_dict)
            missing_keys, unexpected_keys = model.load_state_dict(
                state_dict, strict=False
            )
            all_keys = [i for i, _ in model.named_parameters()]
            num_updated_keys = len(all_keys) - len(missing_keys)
            num_unexpected_keys = len(unexpected_keys)
            print(
                f"{num_updated_keys} parameters are loaded from {pretrained_lora_path}. {num_unexpected_keys} parameters are unexpected."
            )

    def training_step(self, batch, batch_idx):
        # batch["dwpose_data"]/255.: [1, 3, 81, 832, 480], batch["random_ref_dwpose_data"]/255.: [1, 832, 480, 3]
        text, video, path = batch["text"][0], batch["video"], batch["path"][0]
        # 'A person is dancing',  [1, 3, 81, 832, 480], 'data/example_dataset/train/[DLPanda.com][]7309800480371133711.mp4'
        self.pipe_VAE.device = self.device
        dwpose_data = self.dwpose_embedding(
            (
                torch.cat(
                    [
                        batch["dwpose_data"][:, :, :1].repeat(1, 1, 3, 1, 1),
                        batch["dwpose_data"],
                    ],
                    dim=2,
                )
                / 255.0
            ).to(self.device)
        )
        random_ref_dwpose_data = self.randomref_embedding_pose(
            (batch["random_ref_dwpose_data"] / 255.0)
            .to(torch.bfloat16)
            .to(self.device)
            .permute(0, 3, 1, 2)
        ).unsqueeze(
            2
        )  # [1, 20, 104, 60]

        if batch_idx % TB_FERQ == 0:
            # print(
            #     batch["dwpose_data"].shape,
            #     batch["video"].shape,
            #     batch["random_ref_dwpose_data"].shape,
            #     batch["first_frame"].shape,
            #     batch["first_frame"].max(),
            # )
            tensorboard = self.logger.experiment
            input_video_w_pose = (
                batch["dwpose_data"] / 255.0 * 0.5
                + ((0.5 * batch["video"]) + 0.5) * 0.5
            )
            ref_img_w_pose = (
                batch["random_ref_dwpose_data"] / 255.0 * 0.5
                + batch["first_frame"] / 255.0 * 0.5
            )
            tensorboard.add_image("inputs/ref_img", ref_img_w_pose[0].permute(2, 0, 1))
            tensorboard.add_video(
                "inputs/video", input_video_w_pose.permute(0, 2, 1, 3, 4)
            )

        with torch.no_grad():
            if video is not None:
                # prompt
                prompt_emb = self.pipe_VAE.encode_prompt(text)
                # video
                video = video.to(
                    dtype=self.pipe_VAE.torch_dtype, device=self.pipe_VAE.device
                )
                latents = self.pipe_VAE.encode_video(video, **self.tiler_kwargs)[0]
                # image
                if "first_frame" in batch:  # [1, 853, 480, 3]
                    # print(batch["first_frame"].shape)
                    first_frame = Image.fromarray(batch["first_frame"][0].cpu().numpy())
                    _, _, num_frames, height, width = video.shape
                    image_emb = self.pipe_VAE.encode_image(
                        first_frame, num_frames, height, width
                    )
                else:
                    image_emb = {}

                batch = {
                    "latents": latents.unsqueeze(0),
                    "prompt_emb": prompt_emb,
                    "image_emb": image_emb,
                }

        # Data
        p1 = random.random()
        p = random.random()
        if p1 < 0.05:

            dwpose_data = torch.zeros_like(dwpose_data)
            random_ref_dwpose_data = torch.zeros_like(random_ref_dwpose_data)
        latents = batch["latents"].to(self.device)  # [1, 16, 21, 60, 104]
        prompt_emb = batch[
            "prompt_emb"
        ]  # batch["prompt_emb"]["context"]:  [1, 1, 512, 4096]

        prompt_emb["context"] = prompt_emb["context"].to(self.device)
        image_emb = batch["image_emb"]
        if "clip_feature" in image_emb:
            image_emb["clip_feature"] = image_emb["clip_feature"].to(
                self.device
            )  # [1, 257, 1280]
            if p < 0.1:
                image_emb["clip_feature"] = torch.zeros_like(
                    image_emb["clip_feature"]
                )  # [1, 257, 1280]
        if "y" in image_emb:

            if p < 0.1:
                image_emb["y"] = torch.zeros_like(image_emb["y"])
            image_emb["y"] = (
                image_emb["y"].to(self.device) + random_ref_dwpose_data
            )  # [1, 20, 21, 104, 60]

        condition = dwpose_data
        #
        condition = rearrange(condition, "b c f h w -> b (f h w) c").contiguous()
        # Loss
        self.pipe.device = self.device
        noise = torch.randn_like(latents)
        timestep_id = torch.randint(0, self.pipe.scheduler.num_train_timesteps, (1,))
        timestep = self.pipe.scheduler.timesteps[timestep_id].to(
            dtype=self.pipe.torch_dtype, device=self.pipe.device
        )
        extra_input = self.pipe.prepare_extra_input(latents)
        noisy_latents = self.pipe.scheduler.add_noise(latents, noise, timestep)
        training_target = self.pipe.scheduler.training_target(latents, noise, timestep)

        # Compute loss
        noise_pred = self.pipe.denoising_model()(
            noisy_latents,
            timestep=timestep,
            **prompt_emb,
            **extra_input,
            **image_emb,
            use_gradient_checkpointing=self.use_gradient_checkpointing,
            use_gradient_checkpointing_offload=self.use_gradient_checkpointing_offload,
            add_condition=condition,
        )
        loss = torch.nn.functional.mse_loss(noise_pred.float(), training_target.float())
        loss = loss * self.pipe.scheduler.training_weight(timestep)

        # Record log
        self.log("train_loss", loss, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        # trainable_modules = filter(lambda p: p.requires_grad, self.pipe.denoising_model().parameters())
        # optimizer = torch.optim.AdamW(trainable_modules, lr=self.learning_rate)
        # return optimizer
        trainable_modules = [
            {
                "params": filter(
                    lambda p: p.requires_grad, self.pipe.denoising_model().parameters()
                )
            },
            {"params": self.dwpose_embedding.parameters()},
            {"params": self.randomref_embedding_pose.parameters()},
        ]
        optimizer = torch.optim.AdamW(trainable_modules, lr=self.learning_rate)
        return optimizer

    def on_save_checkpoint(self, checkpoint):
        checkpoint.clear()
        # trainable_param_names = list(filter(lambda named_param: named_param[1].requires_grad, self.pipe.denoising_model().named_parameters())) + \
        #                         list(filter(lambda named_param: named_param[1].requires_grad, self.dwpose_embedding.named_parameters())) + \
        #                         list(filter(lambda named_param: named_param[1].requires_grad, self.randomref_embedding_pose.named_parameters()))
        trainable_param_names = list(
            filter(
                lambda named_param: named_param[1].requires_grad,
                self.named_parameters(),
            )
        )

        trainable_param_names = set(
            [named_param[0] for named_param in trainable_param_names]
        )
        # state_dict = self.pipe.denoising_model().state_dict()
        state_dict = self.state_dict()
        # state_dict.update()
        lora_state_dict = {}
        for name, param in state_dict.items():
            if name in trainable_param_names:
                lora_state_dict[name] = param
        checkpoint.update(lora_state_dict)


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--task",
        type=str,
        default="data_process",
        required=True,
        choices=["data_process", "train"],
        help="Task. `data_process` or `train`.",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default=None,
        required=True,
        help="The path of the Dataset.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="./",
        help="Path to save the model.",
    )
    parser.add_argument(
        "--text_encoder_path",
        type=str,
        default="/mnt/data/hnqiu/wanx2.1_t2v/WanX2.1-T2V-14B/models_t5_umt5-xxl-enc-bf16.pth",
        help="Path of text encoder.",
    )
    parser.add_argument(
        "--image_encoder_path",
        type=str,
        default=None,
        help="Path of image encoder.",
    )
    parser.add_argument(
        "--vae_path",
        type=str,
        default="/mnt/data/hnqiu/wanx2.1_t2v/WanX2.1-T2V-14B/WanX2.1_VAE.pth",
        help="Path of VAE.",
    )
    parser.add_argument(
        "--dit_path",
        type=str,
        default=None,
        help="Path of DiT.",
    )
    parser.add_argument(
        "--tiled",
        # default=False,
        default=True,
        action="store_true",
        help="Whether enable tile encode in VAE. This option can reduce VRAM required.",
    )
    parser.add_argument(
        "--tile_size_height",
        type=int,
        default=34,
        help="Tile size (height) in VAE.",
    )
    parser.add_argument(
        "--tile_size_width",
        type=int,
        default=34,
        help="Tile size (width) in VAE.",
    )
    parser.add_argument(
        "--tile_stride_height",
        type=int,
        default=18,
        help="Tile stride (height) in VAE.",
    )
    parser.add_argument(
        "--tile_stride_width",
        type=int,
        default=16,
        help="Tile stride (width) in VAE.",
    )
    parser.add_argument(
        "--steps_per_epoch",
        type=int,
        default=500,
        help="Number of steps per epoch.",
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=80,
        help="Number of frames.",
    )
    # parser.add_argument(
    #     "--height",
    #     type=int,
    #     default=480,
    #     help="Image height.",
    # )
    # parser.add_argument(
    #     "--width",
    #     type=int,
    #     default=832,
    #     help="Image width.",
    # )
    parser.add_argument(
        "--height",
        type=int,
        default=832,
        help="Image height.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=480,
        help="Image width.",
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=1,
        help="Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-5,
        help="Learning rate.",
    )
    parser.add_argument(
        "--accumulate_grad_batches",
        type=int,
        default=1,
        help="The number of batches in gradient accumulation.",
    )
    parser.add_argument(
        "--max_epochs",
        type=int,
        default=1,
        help="Number of epochs.",
    )
    parser.add_argument(
        "--lora_target_modules",
        type=str,
        default="q,k,v,o,ffn.0,ffn.2",
        help="Layers with LoRA modules.",
    )
    parser.add_argument(
        "--init_lora_weights",
        type=str,
        default="kaiming",
        choices=["gaussian", "kaiming"],
        help="The initializing method of LoRA weight.",
    )
    parser.add_argument(
        "--training_strategy",
        type=str,
        default="auto",
        choices=["auto", "deepspeed_stage_1", "deepspeed_stage_2", "deepspeed_stage_3"],
        help="Training strategy",
    )
    parser.add_argument(
        "--lora_rank",
        type=int,
        default=4,
        help="The dimension of the LoRA update matrices.",
    )
    parser.add_argument(
        "--lora_alpha",
        type=float,
        default=4.0,
        help="The weight of the LoRA update matrices.",
    )
    parser.add_argument(
        "--use_gradient_checkpointing",
        default=False,
        action="store_true",
        help="Whether to use gradient checkpointing.",
    )
    parser.add_argument(
        "--use_gradient_checkpointing_offload",
        default=False,
        action="store_true",
        help="Whether to use gradient checkpointing offload.",
    )
    parser.add_argument(
        "--train_architecture",
        type=str,
        default="lora",
        choices=["lora", "full"],
        help="Model structure to train. LoRA training or full training.",
    )
    parser.add_argument(
        "--pretrained_lora_path",
        type=str,
        default=None,
        help="Pretrained LoRA path. Required if the training is resumed.",
    )
    parser.add_argument(
        "--use_swanlab",
        default=False,
        action="store_true",
        help="Whether to use SwanLab logger.",
    )
    parser.add_argument(
        "--swanlab_mode",
        default=None,
        help="SwanLab mode (cloud or local).",
    )
    args = parser.parse_args()
    return args


def data_process(args):
    dataset = TextVideoDataset(
        args.dataset_path,
        os.path.join(args.dataset_path, "metadata.csv"),
        max_num_frames=args.num_frames,
        frame_interval=1,
        num_frames=args.num_frames,
        height=args.height,
        width=args.width,
        is_i2v=args.image_encoder_path is not None,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset, shuffle=False, batch_size=1, num_workers=args.dataloader_num_workers
    )
    model = LightningModelForDataProcess(
        text_encoder_path=args.text_encoder_path,
        image_encoder_path=args.image_encoder_path,
        vae_path=args.vae_path,
        tiled=args.tiled,
        tile_size=(args.tile_size_height, args.tile_size_width),
        tile_stride=(args.tile_stride_height, args.tile_stride_width),
    )
    trainer = pl.Trainer(
        accelerator="gpu",
        devices="auto",
        default_root_dir=args.output_path,
    )
    trainer.test(model, dataloader)


class LightningModelForDataProcess(pl.LightningModule):
    def __init__(
        self,
        text_encoder_path,
        vae_path,
        image_encoder_path=None,
        tiled=False,
        tile_size=(34, 34),
        tile_stride=(18, 16),
    ):
        super().__init__()
        model_path = [text_encoder_path, vae_path]
        if image_encoder_path is not None:
            model_path.append(image_encoder_path)
        model_manager = ModelManager(torch_dtype=torch.bfloat16, device="cpu")
        model_manager.load_models(model_path)
        self.pipe = WanVideoPipeline.from_model_manager(model_manager)

        self.tiler_kwargs = {
            "tiled": tiled,
            "tile_size": tile_size,
            "tile_stride": tile_stride,
        }

    def test_step(self, batch, batch_idx):
        text, video, path = batch["text"][0], batch["video"], batch["path"][0]

        self.pipe.device = self.device
        if video is not None:
            # prompt
            prompt_emb = self.pipe.encode_prompt(text)
            # video
            video = video.to(dtype=self.pipe.torch_dtype, device=self.pipe.device)
            latents = self.pipe.encode_video(video, **self.tiler_kwargs)[0]
            # image
            if "first_frame" in batch:
                first_frame = Image.fromarray(batch["first_frame"][0].cpu().numpy())
                _, _, num_frames, height, width = video.shape
                image_emb = self.pipe.encode_image(
                    first_frame, num_frames, height, width
                )
            else:
                image_emb = {}
            data = {
                "latents": latents,
                "prompt_emb": prompt_emb,
                "image_emb": image_emb,
            }
            torch.save(data, path + ".tensors.pth")


def train_onestage(args):

    dataset = SSTKVideoDataset_onestage(
        **shutterstock_video_dataset,
        max_num_frames=args.num_frames,
        frame_interval=1,
        num_frames=args.num_frames,
        height=args.height,
        width=args.width,
        is_i2v=args.image_encoder_path is not None,
        steps_per_epoch=args.steps_per_epoch,
    )
    # total = len(dataset)
    # not_enough_frames = 0
    # missing_files = 0
    # for index in tqdm(range(len(dataset))):
    #     index = index % len(dataset.video_names_valid)
    #     source_video_name = dataset.video_names_valid[index]

    #     args = (
    #         source_video_name,
    #         dataset.rgb_dir,
    #         dataset.mask_dir,
    #         dataset.smplx_dir,
    #         dataset.trinity_dir,
    #         dataset.pose_dir,
    #         dataset.frame_list_dir,
    #         dataset.num_target + dataset.num_source,
    #     )
    #     video_name, video_data_list, n, status = dataset.process_video(args)
    #     if status != "success":
    #         print(source_video_name, status)
    #         if "missing" in status:
    #             missing_files += 1
    #         if "not enough frames" in status:
    #             not_enough_frames += 1
    # print(
    #     f"missing files {missing_files} / {total}, not enough frames {not_enough_frames} / {total}"
    # )

    dataloader = torch.utils.data.DataLoader(
        dataset, shuffle=True, batch_size=1, num_workers=args.dataloader_num_workers
    )
    model_VAE = LightningModelForDataProcess(
        text_encoder_path=args.text_encoder_path,
        image_encoder_path=args.image_encoder_path,
        vae_path=args.vae_path,
        tiled=args.tiled,
        tile_size=(args.tile_size_height, args.tile_size_width),
        tile_stride=(args.tile_stride_height, args.tile_stride_width),
    )
    model = LightningModelForTrain_onestage(
        dit_path=args.dit_path,
        learning_rate=args.learning_rate,
        train_architecture=args.train_architecture,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_target_modules=args.lora_target_modules,
        init_lora_weights=args.init_lora_weights,
        use_gradient_checkpointing=args.use_gradient_checkpointing,
        use_gradient_checkpointing_offload=args.use_gradient_checkpointing_offload,
        pretrained_lora_path=args.pretrained_lora_path,
        model_VAE=model_VAE,
    )
    if args.use_swanlab:
        from swanlab.integration.pytorch_lightning import SwanLabLogger

        swanlab_config = {"UPPERFRAMEWORK": "DiffSynth-Studio"}
        swanlab_config.update(vars(args))
        swanlab_logger = SwanLabLogger(
            project="wan",
            name="wan",
            config=swanlab_config,
            mode=args.swanlab_mode,
            logdir=os.path.join(args.output_path, "swanlog"),
        )
        logger = [swanlab_logger]
    else:
        time = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        print(
            f"log directory: /checkpoint/avatar/j1wen/tensorboard/sync/UniAnimate-DiT/{time}"
        )
        logger = TensorBoardLogger(
            "/checkpoint/avatar/j1wen/tensorboard/sync/UniAnimate-DiT", name=time
        )
    print("****************start init trainer")

    # print(os.environ)
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator="gpu",
        devices=torch.cuda.device_count(),
        precision="bf16",
        strategy=args.training_strategy,
        # strategy=SingleDeviceStrategy(
        #     device=f"cuda:0",  # or "cuda" if you prefer general
        #     # checkpoint_io=LocalCheckpointIO(),
        # ),
        default_root_dir=args.output_path,
        accumulate_grad_batches=args.accumulate_grad_batches,
        callbacks=[
            pl.callbacks.ModelCheckpoint(
                dirpath=args.output_path,
                save_top_k=-1,
                every_n_train_steps=100,
            )
        ],  # save checkpoints every_n_train_steps
        logger=logger,
        plugins=[LightningEnvironment()],
    )
    print("trainer loaded")

    trainer.fit(model, dataloader)


if __name__ == "__main__":
    args = parse_args()
    if args.task == "data_process":
        data_process(args)
    elif args.task == "train":
        # support VAE and DiT in a single stage
        train_onestage(args)


# lora finetune
# CUDA_VISIBLE_DEVICES="0" python examples/unianimate_wan/train_unianimate_wan.py   --task train   --train_architecture lora --lora_rank 64 --lora_alpha 64  --dataset_path data/example_dataset   --output_path ./models_out_one_GPU   --dit_path "/mnt/user/VideoGeneration_Baselines/Wan2.1/Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00001-of-00007.safetensors,/mnt/user/VideoGeneration_Baselines/Wan2.1/Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00002-of-00007.safetensors,/mnt/user/VideoGeneration_Baselines/Wan2.1/Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00003-of-00007.safetensors,/mnt/user/VideoGeneration_Baselines/Wan2.1/Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00004-of-00007.safetensors,/mnt/user/VideoGeneration_Baselines/Wan2.1/Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00005-of-00007.safetensors,/mnt/user/VideoGeneration_Baselines/Wan2.1/Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00006-of-00007.safetensors,/mnt/user/VideoGeneration_Baselines/Wan2.1/Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00007-of-00007.safetensors"    --max_epochs 10   --learning_rate 1e-4   --accumulate_grad_batches 1   --use_gradient_checkpointing --image_encoder_path "/mnt/user/VideoGeneration_Baselines/Wan2.1/Wan2.1-I2V-14B-720P/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"  --use_gradient_checkpointing_offload
# CUDA_VISIBLE_DEVICES="0,1" python examples/unianimate_wan/train_unianimate_wan.py  --task train   --train_architecture lora --lora_rank 128 --lora_alpha 128  --dataset_path data/example_dataset   --output_path ./models_out   --dit_path "/mnt/user/VideoGeneration_Baselines/Wan2.1/Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00001-of-00007.safetensors,/mnt/user/VideoGeneration_Baselines/Wan2.1/Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00002-of-00007.safetensors,/mnt/user/VideoGeneration_Baselines/Wan2.1/Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00003-of-00007.safetensors,/mnt/user/VideoGeneration_Baselines/Wan2.1/Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00004-of-00007.safetensors,/mnt/user/VideoGeneration_Baselines/Wan2.1/Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00005-of-00007.safetensors,/mnt/user/VideoGeneration_Baselines/Wan2.1/Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00006-of-00007.safetensors,/mnt/user/VideoGeneration_Baselines/Wan2.1/Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00007-of-00007.safetensors"     --max_epochs 10   --learning_rate 1e-4   --accumulate_grad_batches 1   --use_gradient_checkpointing --image_encoder_path "/mnt/user/VideoGeneration_Baselines/Wan2.1/Wan2.1-I2V-14B-720P/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth" --use_gradient_checkpointing_offload --training_strategy "deepspeed_stage_2"
# CUDA_VISIBLE_DEVICES="0,1,2,3" python examples/unianimate_wan/train_unianimate_wan.py  --task train   --train_architecture lora --lora_rank 128 --lora_alpha 128  --dataset_path data/example_dataset   --output_path ./models_out   --dit_path "/mnt/user/VideoGeneration_Baselines/Wan2.1/Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00001-of-00007.safetensors,/mnt/user/VideoGeneration_Baselines/Wan2.1/Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00002-of-00007.safetensors,/mnt/user/VideoGeneration_Baselines/Wan2.1/Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00003-of-00007.safetensors,/mnt/user/VideoGeneration_Baselines/Wan2.1/Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00004-of-00007.safetensors,/mnt/user/VideoGeneration_Baselines/Wan2.1/Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00005-of-00007.safetensors,/mnt/user/VideoGeneration_Baselines/Wan2.1/Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00006-of-00007.safetensors,/mnt/user/VideoGeneration_Baselines/Wan2.1/Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00007-of-00007.safetensors"     --max_epochs 10   --learning_rate 1e-4   --accumulate_grad_batches 1   --use_gradient_checkpointing --image_encoder_path "/mnt/user/VideoGeneration_Baselines/Wan2.1/Wan2.1-I2V-14B-720P/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth" --use_gradient_checkpointing_offload --training_strategy "deepspeed_stage_2"
