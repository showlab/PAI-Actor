
import os
import json
import cv2
from decord import VideoReader
import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F


class ResizeByLongSide:
    def __init__(self, target_long, interpolation=transforms.InterpolationMode.BILINEAR):
        self.target_long = target_long
        self.interpolation = interpolation

    def __call__(self, img):
        w, h = img.size  # PIL.Image.size = (width, height)
        long_side = max(w, h)
        scale = self.target_long / long_side
        new_w = int(round(w * scale))
        new_h = int(round(h * scale))
        return transforms.functional.resize(img, (new_h, new_w), interpolation=self.interpolation, antialias=True)


class CustomTrainDataset(Dataset):
    def __init__(
        self,
        data_jsonl,
        height=512,
        width=512,
        resize_long=1505,
        sample_n_frames=49,
        is_one2three=False,
        training_len=-1,
    ):
        self.training_len = training_len
        self.is_one2three = is_one2three
        self.height = height
        self.width = width
        self.resize_long = resize_long
        self.sample_n_frames = sample_n_frames

        # 从 jsonl 加载所有样本
        self.samples = []
        with open(data_jsonl, 'r') as f:
            for line in f:
                item = json.loads(line.strip())
                self.samples.append(item)

        # video transforms
        self.train_video_transforms = transforms.Compose(
            [
                ResizeByLongSide(self.resize_long),
                transforms.CenterCrop([self.height, self.width]),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.5, 0.5, 0.5],
                    std=[0.5, 0.5, 0.5]
                )
            ]
        )
        # image transforms
        self.train_image_transforms = transforms.Compose(
            [
                ResizeByLongSide(self.resize_long),
                transforms.CenterCrop([self.height, self.width]),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.5, 0.5, 0.5],
                    std=[0.5, 0.5, 0.5]
                )
            ]
        )

    def __len__(self):
        if self.training_len == -1:
            return len(self.samples)
        else:
            return self.training_len

    def __getitem__(self, index):
        index = index % len(self.samples)

        while True:
            sample = self.samples[index]
            first_video_path = sample['control_video']
            third_video_path = sample['video']

            first_video_reader = VideoReader(first_video_path)
            third_video_reader = VideoReader(third_video_path)
            min_frame_length = min(len(first_video_reader), len(third_video_reader))
            if min_frame_length >= self.sample_n_frames:
                break
            else:
                print(f"Too few frames ({min_frame_length})! {first_video_path}")
                index = random.randint(0, len(self.samples) - 1)
                continue

        # ref image: 1张
        ref_image_path = sample['ref_image_path']
        ref_image = Image.open(ref_image_path).convert("RGB")
        ref_pixel_values = self.train_image_transforms(ref_image).unsqueeze(0)  # (1, C, H, W)
        ref_frame_indices = [0]

        # 取前 sample_n_frames 帧
        frame_indices = np.arange(self.sample_n_frames)
        # first video
        first_video = first_video_reader.get_batch(frame_indices).asnumpy() # F, H, W, C
        first_video = [Image.fromarray(frame) for frame in first_video]
        first_pixel_values = torch.stack([self.train_video_transforms(frame) for frame in first_video]) # F, C, H, W
        # third video
        third_video = third_video_reader.get_batch(frame_indices).asnumpy()
        third_video = [Image.fromarray(frame) for frame in third_video]
        third_pixel_values = torch.stack([self.train_video_transforms(frame) for frame in third_video]) # F, C, H, W

        prompt = sample.get('prompt', 'Transform it into the third-person perspective.' if self.is_one2three else 'Transform it into the first-person perspective.')

        return {
            'ref_pixel_values': ref_pixel_values, # (1, C, H, W)
            'ref_frame_indices': ref_frame_indices if self.is_one2three else [],
            'first_pixel_values': first_pixel_values.permute(1, 0, 2, 3), # C, F, H, W
            'third_pixel_values': third_pixel_values.permute(1, 0, 2, 3), # C, F, H, W
            'prompts': prompt,
        }


class CustomTestDataset(Dataset):
    def __init__(
        self,
        original_video_root=None,
        ref_image_root=None,
        prompt_file=None,
        data_jsonl=None,
        height=512,
        width=512,
        resize_long=1505,
        sample_n_frames=49,
        stride=1,
        is_one2three=False,
        training_len=-1,
    ):
        self.stride = stride
        self.training_len = training_len
        self.is_one2three = is_one2three
        self.height = height
        self.width = width
        self.resize_long = resize_long
        self.sample_n_frames = sample_n_frames

        # JSONL 模式：直接从 jsonl 读取所有信息
        self.use_jsonl = data_jsonl is not None
        if self.use_jsonl:
            self.samples = []
            with open(data_jsonl, 'r') as f:
                for line in f:
                    self.samples.append(json.loads(line.strip()))
        else:
            # 旧的目录模式
            self.original_video_root = original_video_root
            if self.is_one2three:
                self.ref_image_root = ref_image_root

            self.prompt_dict = {}
            self.line_indices = []
            if prompt_file and os.path.exists(prompt_file):
                with open(prompt_file, 'r') as f:
                    for file_line in f:
                        item = json.loads(file_line.strip())
                        line_idx = item.get('line', None)
                        caption = item.get('caption', '')
                        if line_idx is not None:
                            self.prompt_dict[line_idx] = caption
                            self.line_indices.append(line_idx)

            self.original_video_dict = {}
            for video in os.listdir(self.original_video_root):
                video_path = os.path.join(self.original_video_root, video)
                video_idx = int(os.path.splitext(video)[0])
                self.original_video_dict[video_idx] = video_path

            if self.is_one2three:
                self.ref_image_dict = {}
                for d in os.listdir(self.ref_image_root):
                    dir_path = os.path.join(self.ref_image_root, d)
                    if os.path.isdir(dir_path):
                        dir_idx = int(d)
                        self.ref_image_dict[dir_idx] = dir_path

            self.original_video_paths = [os.path.join(self.original_video_root, video) for video in sorted(os.listdir(self.original_video_root))]
            if self.is_one2three:
                self.ref_image_paths = [os.path.join(self.ref_image_root, d) for d in sorted(os.listdir(self.ref_image_root)) if os.path.isdir(os.path.join(self.ref_image_root, d))]
            self.original_video_len = len(self.original_video_paths)
            if self.is_one2three:
                self.ref_image_len = len(self.ref_image_paths)

        # video transforms
        self.test_video_transforms = transforms.Compose(
            [
                ResizeByLongSide(self.resize_long),
                transforms.CenterCrop([self.height, self.width]),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.5, 0.5, 0.5],
                    std=[0.5, 0.5, 0.5]
                )
            ]
        )
        if self.is_one2three:
            self.test_image_transforms = transforms.Compose(
                [
                    ResizeByLongSide(self.resize_long),
                    transforms.CenterCrop([self.height, self.width]),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.5, 0.5, 0.5],
                        std=[0.5, 0.5, 0.5]
                    )
                ]
            )

    def __len__(self):
        if self.use_jsonl:
            return len(self.samples)
        if self.training_len == -1:
            if self.line_indices:
                return len(self.line_indices)
            return self.original_video_len
        else:
            return self.training_len

    def __getitem__(self, index):
        if self.use_jsonl:
            return self._getitem_jsonl(index)
        return self._getitem_legacy(index)

    def _getitem_jsonl(self, index):
        sample = self.samples[index]
        control_video_path = sample['control_video']
        prompt = sample.get('prompt', '')

        # ref image
        if self.is_one2three and 'ref_image_path' in sample:
            ref_image = Image.open(sample['ref_image_path']).convert("RGB")
            ref_pixel_values = self.test_image_transforms(ref_image).unsqueeze(0)  # (1, C, H, W)
            ref_frame_indices = [0]
        else:
            ref_pixel_values = []
            ref_frame_indices = []

        # 读取 control video
        video_reader = VideoReader(control_video_path)
        frame_indices = np.arange(min(self.sample_n_frames, len(video_reader)))
        input_frames = video_reader.get_batch(frame_indices).asnumpy()
        input_frame_list = [Image.fromarray(frame) for frame in input_frames]
        input_pixel_values = torch.stack([self.test_video_transforms(frame) for frame in input_frame_list])

        # 用 index 作为文件名
        path = str(index)

        return {
            'ref_pixel_values': ref_pixel_values,
            'ref_frame_indices': ref_frame_indices if self.is_one2three else [],
            'input_pixel_values': input_pixel_values.permute(1, 0, 2, 3),
            'prompts': prompt,
            'path': path,
        }

    def _getitem_legacy(self, index):
        if self.line_indices:
            index = index % len(self.line_indices)
            line_idx = self.line_indices[index]
        else:
            index = index % self.original_video_len
            line_idx = index

        if self.is_one2three:
            if self.line_indices and line_idx in self.ref_image_dict:
                ref_image_dir = self.ref_image_dict[line_idx]
            else:
                ref_image_dir = self.ref_image_paths[index]
            all_ref_files = sorted(os.listdir(ref_image_dir))[:3]
            ref_images = []
            ref_frame_indices = []
            for filename in all_ref_files:
                try:
                    frame_idx = int(filename.split('_')[-1].split('.')[0])
                    ref_frame_indices.append(frame_idx)
                    ref_images.append(Image.open(os.path.join(ref_image_dir, filename)).convert("RGB"))
                except:
                    continue
            if len(ref_images) > 0:
                ref_pixel_values = torch.stack([self.test_image_transforms(img) for img in ref_images])
            else:
                ref_pixel_values = []
                ref_frame_indices = []
        else:
            ref_pixel_values = []
            ref_frame_indices = []

        if self.line_indices and line_idx in self.original_video_dict:
            original_video_path = self.original_video_dict[line_idx]
        else:
            original_video_path = self.original_video_paths[index]

        video_reader = VideoReader(original_video_path)
        frame_length = len(video_reader)
        frame_indices = np.arange(self.sample_n_frames)
        input_frames = video_reader.get_batch(frame_indices).asnumpy()
        input_frame_list = [Image.fromarray(frame) for frame in input_frames]
        input_pixel_values = torch.stack([self.test_video_transforms(frame) for frame in input_frame_list])

        path = str(line_idx)

        if line_idx in self.prompt_dict:
            prompt = self.prompt_dict[line_idx]
        else:
            prompt = 'Transform it into the third-person perspective.' if self.is_one2three else 'Transform it into the first-person perspective.'

        return {
            'ref_pixel_values': ref_pixel_values,
            'ref_frame_indices': ref_frame_indices if self.is_one2three else [],
            'input_pixel_values': input_pixel_values.permute(1, 0, 2, 3),
            'prompts': prompt,
            'path': path,
        }


# for debug
if __name__ == '__main__':
    my_dataset = CustomTrainDataset(
        video_root='your-path1',
        video_root2='your-path2',
        first_video_root='your-path3',
        height=480,
        width=720,
        sample_n_frames=81,
    )
    my_dataloader = DataLoader(my_dataset,
                               batch_size=1,
                               shuffle=False)
    #
    for batch in my_dataset:
        breakpoint()
