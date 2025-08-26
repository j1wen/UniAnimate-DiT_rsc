# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

import logging

import os
from typing import List
import shutil

import torch
from iopath.common.file_io import PathManager, PathManagerFactory
from PIL import Image
from PIL.Image import Image as PILImageType
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from scripts.qwen.qwen_utils import process_vision_info
import cv2
from tqdm import tqdm
import sys


# os.environ["FLASH_ATTENTION_TRITON_AMD_ENABLE"] = (
#     "TRUE"  # need to use flash attention 2
# )
logger: logging.Logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
path_manager: PathManager = PathManagerFactory.get(defaults_setup=True)


class Qwen2_5_VL:
    def __init__(self, model_dir: str) -> None:
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_dir,
            torch_dtype=torch.bfloat16,
            # flash-attn package is out-dated and cannot be used in latest transformers-stack, further version upgrade is needed
            attn_implementation="sdpa",
            device_map="cuda",
        ).eval()
        self.processor = AutoProcessor.from_pretrained(model_dir, device_map="cuda")

    def image_query(
        self,
        images: List[PILImageType],
        prompt: str,
        max_new_tokens: int = 128,
    ) -> str:
        messages = [
            {"role": "system", "content": [{"type": "text", "text": prompt}]},
            {
                "role": "user",
                "content": [{"type": "image", "image": image} for image in images],
            },
        ]
        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device, dtype=torch.bfloat16)
        input_len = inputs["input_ids"].shape[-1]

        with torch.inference_mode():
            generation = self.model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False
            )
            generation = generation[0][input_len:]

        decoded = self.processor.decode(generation, skip_special_tokens=True)
        return decoded

    def video_query(
        self,
        video_complex,
        prompt: str,
        max_new_tokens: int = 128,
    ) -> str:
        """
        video_path: path to a video file
        prompt: prompt to generate the caption
        """
        messages = [
            {"role": "system", "content": [{"type": "text", "text": prompt}]},
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": video_complex,
                    }
                ],
            },
        ]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        if torch.cuda.device_count() == 1:
            for key in inputs:
                if isinstance(inputs[key], torch.Tensor):
                    inputs[key] = inputs[key].cuda()
        with torch.inference_mode():
            generated_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return output_text


def name2parts(cid):
    if len(cid) > 4:
        parts = [cid[i : i + 4] for i in range(0, len(cid), 4)]
        cid_parts = "/".join(parts)
    else:
        cid_parts = cid
    return cid_parts


if __name__ == "__main__":
    # model_dir = "manifold://frl_gemini_body_shape_and_pose/tree/auto_critic/Qwen2.5-VL-7B-Instruct/"
    # local_model_dir = path_manager.get_local_path(model_dir)
    # print(local_model_dir)
    model = Qwen2_5_VL("/home/j1wen/rsc/UniAnimate-DiT/scripts/qwen/Qwen2.5-VL-7B-Instruct")

    # # Example of image query
    # image_fp = "manifold://frl_gemini_body_shape_and_pose/tree/dataset/truetony/RGB/truetonydgx_003000.png"
    # local_image_fp = path_manager.get_local_path(image_fp)
    # image = Image.open(local_image_fp)
    # prompt = (
    #     "Tell me whether the attached image is a real photo or a synthetic rendering"
    # )
    # print(model.image_query([image], prompt))
    # Example of video query
    # video_fp = (
    #     "manifold://frl_gemini_body_shape_and_pose/tree/temp/lighticon_example.mp4"
    # )
    with open("data/example_dataset/SSTK350K/train_scenes.txt") as f:
        video_names = [line.strip() for line in f.readlines()]
    image_root = "/decoders/suzhaoen/legion/lhm/resampled/full_res_images"

    done_list = os.listdir("/checkpoint/avatar/j1wen/loose_done_list")

    static_video_names = []
    for video_name in tqdm(video_names[int(sys.argv[1]):int(sys.argv[1]) + 10000]):
        print(video_name)
        if video_name in done_list:
            continue
        try:
            video_name_parts = name2parts(video_name)
            image_dir = os.path.join(image_root, video_name_parts)

            tmp_video_name = f'/home/j1wen/rsc_unsync/temp_{int(sys.argv[1]):06d}.mp4'

            frame = cv2.imread(os.path.join(image_dir, sorted(os.listdir(image_dir))[0]))
            height, width, layers = frame.shape

            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video = cv2.VideoWriter(tmp_video_name, fourcc, 30, (width,height))

            common_names = []
            for file_name in sorted(os.listdir(image_dir)):
                if file_name.endswith(".jpg") or file_name.endswith(".png"):
                    common_names.append(os.path.join(image_dir, file_name))
                    video.write(cv2.imread(os.path.join(image_dir, file_name)))
            video.release()

            # prompt = "Describe the video content in details. Answer Yes or No: is it a video of a person on a pure-color background, e.g., green, black or white?"
            prompt = """Analyze the video and determine if the following conditions are met:
    1) The person in the video is wearing loose dresses or long coats.
    2) The person is performing large motions, with particular emphasis on noticeable movements of the feet and legs.
    Please provide a yes/no answer for each condition along with a brief explanation or evidence from the video frames."""
            ans = model.video_query(tmp_video_name, prompt)[0]
            print(ans)
            # print(ans)
            if ans.lower().count("yes") >= 2:
                static_video_names.append(video_name)
                shutil.copyfile(tmp_video_name, "/checkpoint/avatar/j1wen/loose/" + video_name + '.mp4')
                # if len(static_video_names) > 900:
                #     break
            with open(f"/checkpoint/avatar/j1wen/loose_done_list/{video_name}", "w") as f:
                f.write("")
        except:
            print(f"video {video_name} failed")

    # local_video_fp = "/home/j1wen/rsc/UniAnimate-DiT/scripts/qwen/lighticon_example.mp4"
    # prompt = "Describe the video content in details. Answer Yes or No in the last word: is it from a static camera?"
    # print(model.video_query(local_video_fp, prompt)[0])
