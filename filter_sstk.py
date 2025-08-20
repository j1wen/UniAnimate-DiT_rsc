import argparse, imageio, os, torch
import contextlib
import copy
import glob
import gzip
import json
import pickle
import random
import sys
from io import BytesIO

import cv2

import numpy as np
import pandas as pd
import pymomentum.geometry as pym_geometry
import pymomentum.models as pym_models
import pymomentum.skel_state as pym_skel_state
import pytorch_lightning as pl

import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from diffsynth import (
    load_state_dict,
    load_state_dict_from_folder,
    ModelManager,
    save_video,
    WanVideoPipeline,
)
from einops import rearrange

from examples.unianimate_wan.train_util import coco_wholebody2openpose, draw_keypoints
from peft import inject_adapter_in_model, LoraConfig
from PIL import Image, ImageFilter

from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.plugins.environments import LightningEnvironment
from pytorch_lightning.strategies import SingleDeviceStrategy
from torchvision.transforms import v2
from tqdm import tqdm

os.environ["TOKENIZERS_PARALLELISM"] = "false"

TB_FREQ = 100


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
    keypoints_root="/decoders/matthewhu/itw_body_tracking_ls/delivery/sstk_350k/outputs",
    smplx_root="/decoders/junxuanli/legion/lhm/resampled/full_res_images/smplx_params",  # not used
    trinity_root="/decoders/matthewhu/itw_body_tracking_ls/delivery/sstk_350k/outputs",  # used
    mask_root="/decoders/matthewhu/itw_body_tracking_ls/delivery/sstk_350k/outputs",
    index_root="/xrcia_shared/ariyanzarei/filtering/SSTK_350K/results/sstk_350k/final_filtered_indices/final_subject_list_gt_8.txt",
    frame_list_root="/xrcia_shared/ariyanzarei/filtering/SSTK_350K/results/sstk_350k/final_filtered_indices/valid_frames",
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

        self.trinity = pym_models.load_default_trinity(
            model_params_type=pym_models.ModelParams.Compact,
            model_definition_version=(5, 0),
        )
        local_input_path_dir = "DVB665_flipped.glb"
        [self.char, poses, offsets, fps] = pym_geometry.Character.load_gltf_with_motion(
            local_input_path_dir
        )

        trinity_coco_regressor_path = "trinity_v3_S0_full_coco_regressor_compress.npz"
        with open(trinity_coco_regressor_path, "rb") as f:
            data = np.load(f)
            self.trinity_coco_regressor = data["trinity2coco"]
            self.joint_names = data["joint_names"]

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
        mask_video_dir = os.path.join(
            mask_dir, video_name_parts, "pre_processing/segmentation"
        )
        smplx_video_dir = os.path.join(smplx_dir, video_name_parts, "smplx_params")
        if not os.path.exists(smplx_video_dir):
            smplx_video_dir = os.path.join(smplx_dir, video_name_parts)
        trinity_video_dir = os.path.join(
            trinity_dir, video_name_parts, "ik_refinement/out.pkl"
        )
        pose_video_dir = os.path.join(
            pose_dir, video_name_parts, "pre_processing/out.pkl"
        )
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
            return video_name, [], 0, [], "missing path: " + ",".join(missing_path)

        with open(frame_list_file, "r") as f:
            all_images_file = f.readlines()
            valid_common_names = [x.strip() for x in all_images_file]

        # Collect file names (without extensions)
        common_names = []
        for file_name in sorted(os.listdir(image_dir)):
            if file_name.endswith(".jpg") or file_name.endswith(".png"):
                file_name = file_name.split(".")[0]
                common_names.append(file_name)

        rgb_extension = os.listdir(image_dir)[0].split(".")[-1]
        assert rgb_extension == "jpg" or rgb_extension == "png"

        # Build the list of dictionaries for this video
        poses = pickle.load(open(trinity_video_dir, "rb"))

        valid_video_data_list = []
        for name in valid_common_names:
            frame_name = int(name.split("_")[-1])
            name = f"{video_name}_{frame_name:06d}"
            data_info = {
                "rgb_path": os.path.join(image_dir, name + "." + rgb_extension),
                "mask_path": os.path.join(mask_video_dir, name + ".png"),
                "pose": poses["pose_params"][name],
                "K": poses["K"][name],
                "Rt": poses["Rt"][name],
            }
            data_info["video_name"] = video_name
            valid_video_data_list.append(data_info)

        video_data_list = []
        for name in common_names:
            data_info = {
                "rgb_path": os.path.join(image_dir, name + "." + rgb_extension),
                "mask_path": os.path.join(mask_video_dir, name + ".png"),
                # "smplx_path": os.path.join(smplx_video_dir, name + ".json"),
                # "trinity_path": os.path.join(trinity_video_dir, name + ".txt"),
                "pose": poses["pose_params"][name],
                "K": poses["K"][name],
                "Rt": poses["Rt"][name],
            }
            data_info["video_name"] = video_name
            video_data_list.append(data_info)

        # Ensure there are enough samples in this video
        if len(video_data_list) < num_target:
            return (
                video_name,
                [],
                0,
                [],
                "not enough frame: " + str(len(video_data_list)),
            )

        return (
            video_name,
            video_data_list,
            len(video_data_list),
            valid_video_data_list,
            "success",
        )

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
        _, video_data_list, num_data_list, valid_video_data_list, _ = (
            self.process_video(args)
        )
        if len(valid_video_data_list) == 0:
            return None

        source_idxs = random.sample(range(len(valid_video_data_list)), self.num_source)
        if self.load_face:
            source_idxs_face = random.sample(
                range(len(valid_video_data_list)), self.num_source
            )
        source_data_list = []
        for i in range(len(source_idxs)):
            source_idx = source_idxs[i]
            source_data_info = copy.deepcopy(valid_video_data_list[source_idx])
            source_data = self.get_data_info_helper(
                source_data_info, video_shape_params=None, with_face=False
            )

            if self.load_face:
                assert False
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
                assert False
                face_expr_image = source_data["face_img"]
                source_data["face_img"] = source_data_face["face_img"]
                source_data["face_expr_image"] = face_expr_image

            source_data_list.append(source_data)

        ## randomly sample num_target target images
        # random sample fps
        stride = random.randint(1, self.sample_fps)

        _total_frame_num = num_data_list
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
                target_data_info, video_shape_params=None, with_face=True
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
        img = cv2.imread(data_info["rgb_path"])[
            :, :, [2, 1, 0]
        ]  ## bgr image is default
        mask = cv2.imread(data_info["mask_path"])

        mask = (mask[:, :, 0] > 0).astype(np.uint8) * 255

        cond_img = img

        # body_pose_params = json.load(open(data_info["smplx_path"]))

        ## coco 133 wholebody keypoints, pixel aligned
        pose = torch.tensor(data_info["pose"])
        skel_states = pym_geometry.model_parameters_to_skeleton_state(
            self.char, pose[:204]
        )

        # COCO poses
        joints3d = skel_states[:, :3]
        skinned_verts = self.trinity.skin_points(skel_states)
        coco_keypoints3d = torch.from_numpy(self.trinity_coco_regressor).to(
            torch.float32
        ) @ torch.cat([joints3d, skinned_verts], dim=0)
        openpose_keypoints3d = coco_wholebody2openpose(coco_keypoints3d)

        openpose_keypoints2d = np.concatenate(
            [openpose_keypoints3d, np.ones_like(openpose_keypoints3d[:, :1])], axis=-1
        )
        openpose_keypoints2d = data_info["Rt"] @ openpose_keypoints2d.T
        openpose_keypoints2d = (data_info["K"] @ openpose_keypoints2d).T
        openpose_keypoints2d = openpose_keypoints2d[:, :2] / openpose_keypoints2d[:, 2:]

        H, W = img.shape[:2]
        canvas_without_face, canvas = draw_keypoints(openpose_keypoints2d, (H, W))

        video_name = data_info["video_name"]

        if img is None:
            return None

        data_info = {
            "img": img,
            "body_img": Image.fromarray(cond_img),
            "img_id": os.path.basename(data_info["rgb_path"]),
            "img_path": data_info["rgb_path"],
            "pose_img": Image.fromarray(canvas_without_face),
            "mask": mask,  # shape: H x W (uint8, with 255 being fg mask, 0 being bg mask, 128 being padded mask)
            "video_name": video_name,
        }

        return data_info

    def __getitem__(self, index):
        index = index % len(self.video_names_valid)
        data = self.get_data_info(index)

        success = False
        frame_list = []
        dwpose_list = []
        mask_list = []
        # try:
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
            target_crop_width = int(l_hight * self.width / self.height)
            random_center_y, random_center_x = np.median(
                np.stack(random_ref_mask.nonzero()), axis=1
            )
            if random_center_x < target_crop_width // 2:
                random_center_x = target_crop_width // 2
            if random_center_x + target_crop_width // 2 > l_width:
                random_center_x = l_width - target_crop_width // 2
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
                [
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
                            self.resize(ss.crop((x1, y1, l_width - x2, l_hight - y2)))
                        )
                    ).permute(2, 0, 1)
                    for ss in dwpose_list
                ],
                dim=0,
            )

        video_data = torch.zeros(
            self.max_frames, 3, self.misc_size[0], self.misc_size[1]
        )
        dwpose_data = torch.zeros(
            self.max_frames, 3, self.misc_size[0], self.misc_size[1]
        )

        if have_frames:
            video_data[: len(frame_list), ...] = video_data_tmp

            dwpose_data[: len(frame_list), ...] = dwpose_data_tmp

        video_data = video_data.permute(1, 0, 2, 3)
        dwpose_data = dwpose_data.permute(1, 0, 2, 3)

        caption = "a person is dancing"
        # except Exception as e:
        #     #
        #     caption = "a person is dancing"
        #     #
        #     video_data = torch.zeros(
        #         3, self.max_frames, self.misc_size[0], self.misc_size[1]
        #     )
        #     random_ref_frame_tmp = torch.zeros(
        #         self.misc_size[0], self.misc_size[1], 3
        #     ).int()
        #     vit_image = torch.zeros(3, self.misc_size[0], self.misc_size[1])

        #     dwpose_data = torch.zeros(
        #         3, self.max_frames, self.misc_size[0], self.misc_size[1]
        #     )
        #     #
        #     random_ref_dwpose_tmp = torch.zeros(self.misc_size[0], self.misc_size[1], 3)
        #     print(
        #         "{} read video frame failed with error: {}".format(
        #             "".join(self.video_names_valid[index]), e
        #         )
        #     )

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


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--task",
        type=str,
        default="data_process",
        choices=["data_process", "train"],
        help="Task. `data_process` or `train`.",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default=None,
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
        default=81,
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
    valid_scenes = []
    with open("/home/j1wen/rsc_unsync/valid_scenes.txt", "r") as f:
        valid_scenes = [line.strip() for line in f.readlines()]
    val_scenes = random.choices(valid_scenes, k=100)
    train_scenes = [scene for scene in valid_scenes if scene not in val_scenes]
    with open("/home/j1wen/rsc_unsync/val_scenes.txt", "w") as f:
        f.write("\n".join(val_scenes))
    with open("/home/j1wen/rsc_unsync/train_scenes.txt", "w") as f:
        f.write("\n".join(train_scenes))
    print("done!")

    # dataset = SSTKVideoDataset_onestage(
    #     **shutterstock_video_dataset_v2,
    #     max_num_frames=args.num_frames,
    #     frame_interval=1,
    #     num_frames=args.num_frames,
    #     height=args.height,
    #     width=args.width,
    #     is_i2v=True,
    #     steps_per_epoch=args.steps_per_epoch,
    # )

    # invalid_scenes = []
    # valid_scenes = []
    # for scene_name in tqdm(dataset.video_names_valid):
    #     video_name_parts = dataset.name2parts(scene_name)
    #     image_dir = os.path.join(dataset.data_root, video_name_parts)
    #     if len(os.listdir(image_dir)) < 82:
    #         print(scene_name)
    #         invalid_scenes.append(scene_name)
    #     else:
    #         valid_scenes.append(scene_name)
    # with open("/home/j1wen/rsc_unsync/invalid_scenes.txt", "w") as f:
    #     f.write("\n".join(invalid_scenes))
    # with open("/home/j1wen/rsc_unsync/valid_scenes.txt", "w") as f:
    #     f.write("\n".join(valid_scenes))

    # total = len(dataset)
    # not_enough_frames = 0
    # missing_files = 0
    # for i in range(total):
    #     data = dataset[i]
    #     input_video_w_pose = (
    #         data["dwpose_data"] / 255.0 * 0.5 + ((0.5 * data["video"]) + 0.5) * 0.5
    #     )
    #     ref_img_w_pose = (
    #         data["random_ref_dwpose_data"] / 255.0 * 0.5
    #         + data["first_frame"] / 255.0 * 0.5
    #     )
    #     input_video_w_pose = (
    #         (input_video_w_pose * 255.0).detach().cpu().numpy().astype(np.uint8)
    #     )
    #     ref_img_w_pose = (
    #         (ref_img_w_pose * 255.0).detach().cpu().numpy().astype(np.uint8)
    #     )
    #     video_out = [Image.fromarray(ref_img_w_pose)]
    #     for frame_id in range(input_video_w_pose.shape[1]):
    #         video_out.append(
    #             Image.fromarray(
    #                 input_video_w_pose[:, frame_id, :, :].transpose(1, 2, 0)
    #             )
    #         )

    #     save_video(
    #         video_out,
    #         "{}/video_480P_{}.mp4".format(
    #             "/home/j1wen/rsc_unsync/vis_sstk350k", data["path"]
    #         ),
    #         fps=15,
    #         quality=5,
    #     )
    # print(
    #     f"missing files {missing_files} / {total}, not enough frames {not_enough_frames} / {total}"
    # )


if __name__ == "__main__":
    args = parse_args()
    data_process(args)


# lora finetune
# CUDA_VISIBLE_DEVICES="0" python examples/unianimate_wan/train_unianimate_wan.py   --task train   --train_architecture lora --lora_rank 64 --lora_alpha 64  --dataset_path data/example_dataset   --output_path ./models_out_one_GPU   --dit_path "/mnt/user/VideoGeneration_Baselines/Wan2.1/Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00001-of-00007.safetensors,/mnt/user/VideoGeneration_Baselines/Wan2.1/Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00002-of-00007.safetensors,/mnt/user/VideoGeneration_Baselines/Wan2.1/Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00003-of-00007.safetensors,/mnt/user/VideoGeneration_Baselines/Wan2.1/Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00004-of-00007.safetensors,/mnt/user/VideoGeneration_Baselines/Wan2.1/Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00005-of-00007.safetensors,/mnt/user/VideoGeneration_Baselines/Wan2.1/Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00006-of-00007.safetensors,/mnt/user/VideoGeneration_Baselines/Wan2.1/Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00007-of-00007.safetensors"    --max_epochs 10   --learning_rate 1e-4   --accumulate_grad_batches 1   --use_gradient_checkpointing --image_encoder_path "/mnt/user/VideoGeneration_Baselines/Wan2.1/Wan2.1-I2V-14B-720P/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"  --use_gradient_checkpointing_offload
# CUDA_VISIBLE_DEVICES="0,1" python examples/unianimate_wan/train_unianimate_wan.py  --task train   --train_architecture lora --lora_rank 128 --lora_alpha 128  --dataset_path data/example_dataset   --output_path ./models_out   --dit_path "/mnt/user/VideoGeneration_Baselines/Wan2.1/Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00001-of-00007.safetensors,/mnt/user/VideoGeneration_Baselines/Wan2.1/Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00002-of-00007.safetensors,/mnt/user/VideoGeneration_Baselines/Wan2.1/Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00003-of-00007.safetensors,/mnt/user/VideoGeneration_Baselines/Wan2.1/Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00004-of-00007.safetensors,/mnt/user/VideoGeneration_Baselines/Wan2.1/Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00005-of-00007.safetensors,/mnt/user/VideoGeneration_Baselines/Wan2.1/Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00006-of-00007.safetensors,/mnt/user/VideoGeneration_Baselines/Wan2.1/Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00007-of-00007.safetensors"     --max_epochs 10   --learning_rate 1e-4   --accumulate_grad_batches 1   --use_gradient_checkpointing --image_encoder_path "/mnt/user/VideoGeneration_Baselines/Wan2.1/Wan2.1-I2V-14B-720P/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth" --use_gradient_checkpointing_offload --training_strategy "deepspeed_stage_2"
# CUDA_VISIBLE_DEVICES="0,1,2,3" python examples/unianimate_wan/train_unianimate_wan.py  --task train   --train_architecture lora --lora_rank 128 --lora_alpha 128  --dataset_path data/example_dataset   --output_path ./models_out   --dit_path "/mnt/user/VideoGeneration_Baselines/Wan2.1/Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00001-of-00007.safetensors,/mnt/user/VideoGeneration_Baselines/Wan2.1/Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00002-of-00007.safetensors,/mnt/user/VideoGeneration_Baselines/Wan2.1/Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00003-of-00007.safetensors,/mnt/user/VideoGeneration_Baselines/Wan2.1/Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00004-of-00007.safetensors,/mnt/user/VideoGeneration_Baselines/Wan2.1/Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00005-of-00007.safetensors,/mnt/user/VideoGeneration_Baselines/Wan2.1/Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00006-of-00007.safetensors,/mnt/user/VideoGeneration_Baselines/Wan2.1/Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00007-of-00007.safetensors"     --max_epochs 10   --learning_rate 1e-4   --accumulate_grad_batches 1   --use_gradient_checkpointing --image_encoder_path "/mnt/user/VideoGeneration_Baselines/Wan2.1/Wan2.1-I2V-14B-720P/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth" --use_gradient_checkpointing_offload --training_strategy "deepspeed_stage_2"
