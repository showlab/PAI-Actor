"""
Face-conditioned dataset for the channel-concat + face-adapter trainer.

Extends CustomTrainDataset: in addition to control/target/ref videos it loads, per
clip, the N per-person face crops (src_face*.mp4, 512x512, frame-aligned to the
TARGET video) and builds per-person SPATIAL masks at the latent grid from
face_tracks.json bboxes (computed on the fly, NOT stored).

jsonl is data_all_face.jsonl (produced by face_crop/crop_faces.py): each record has
the base fields plus `face_dir` and `num_persons`.

Masks: face_tracks.json crop_bboxes are in ORIGINAL `video` pixel coords. The target
frames go through ResizeByLongSide(resize_long) + CenterCrop(height,width), so the
same transform is applied to the bbox before rasterizing to the latent grid
(height//16, width//16). Temporal length = sample_n_frames (pixel frames); the
transformer temporally pools masks down to the latent frame count.

Variable N across clips -> use face_collate_fn to zero-pad to the batch max
(zero mask => that padded person contributes nothing).
"""
import os, json, math
import numpy as np
import torch
from PIL import Image
from decord import VideoReader

from datasets.custom_dataset import CustomTrainDataset

# Wan2.2 VAE spatial compression (16) * DiT patch (2) -> token grid = pixels / 32
LATENT_DIV = 32


class CustomTrainDatasetFace(CustomTrainDataset):
    def __init__(self, *args, face_size=512, max_persons=0, **kwargs):
        super().__init__(*args, **kwargs)
        self.face_size = face_size
        self.max_persons = max_persons  # 0 = no cap

    def _load_face_video(self, path, frame_indices):
        vr = VideoReader(path)
        n = len(vr)
        idx = [min(i, n - 1) for i in frame_indices]
        frames = vr.get_batch(idx).asnumpy()  # F,H,W,C uint8
        t = torch.from_numpy(frames).float().permute(3, 0, 1, 2) / 127.5 - 1.0  # C,F,H,W in [-1,1]
        if t.shape[-1] != self.face_size or t.shape[-2] != self.face_size:
            t = torch.nn.functional.interpolate(
                t.permute(1, 0, 2, 3), size=(self.face_size, self.face_size),
                mode="bilinear", align_corners=False).permute(1, 0, 2, 3)
        return t  # C,F,H,W

    def _bbox_to_latent(self, bbox, W0, H0):
        """orig-pixel (x1,y1,x2,y2) -> latent-grid (gx1,gy1,gx2,gy2) after the same
        ResizeByLongSide(resize_long)+CenterCrop(height,width) as the target frames."""
        scale = self.resize_long / max(W0, H0)
        nw, nh = round(W0 * scale), round(H0 * scale)
        ox = (nw - self.width) / 2.0
        oy = (nh - self.height) / 2.0
        x1, y1, x2, y2 = bbox
        x1c, x2c = x1 * scale - ox, x2 * scale - ox
        y1c, y2c = y1 * scale - oy, y2 * scale - oy
        w_lat, h_lat = self.width // LATENT_DIV, self.height // LATENT_DIV
        gx1 = int(np.clip(math.floor(x1c / LATENT_DIV), 0, w_lat))
        gx2 = int(np.clip(math.ceil(x2c / LATENT_DIV), 0, w_lat))
        gy1 = int(np.clip(math.floor(y1c / LATENT_DIV), 0, h_lat))
        gy2 = int(np.clip(math.ceil(y2c / LATENT_DIV), 0, h_lat))
        return gx1, gy1, gx2, gy2

    def _build_masks(self, tracks, frame_indices):
        """-> [N, T_pixel, h_lat, w_lat] float {0,1}."""
        W0, H0 = tracks["frame_size"]
        n_px = tracks["n_frames"]
        persons = tracks["persons"]
        T = len(frame_indices)
        w_lat, h_lat = self.width // LATENT_DIV, self.height // LATENT_DIV
        masks = torch.zeros(len(persons), T, h_lat, w_lat)
        for pi, person in enumerate(persons):
            boxes = person["crop_bboxes"]
            present = person.get("present")          # per original frame, 1 = detected
            for ti, f in enumerate(frame_indices):
                f = min(f, n_px - 1)
                # per-frame valid: no DETECTED face that frame -> no injection
                # (crop_bboxes are gap-filled by interpolation, but `present`
                #  marks the frames that actually had a face; honour that).
                if present is not None and not present[f]:
                    continue
                gx1, gy1, gx2, gy2 = self._bbox_to_latent(boxes[f], W0, H0)
                if gx2 > gx1 and gy2 > gy1:
                    masks[pi, ti, gy1:gy2, gx1:gx2] = 1.0
        return masks

    def __getitem__(self, index):
        # robust: a corrupt/truncated control or face video (e.g. failed mesh
        # render, "moov atom not found") would otherwise crash the dataloader
        # worker and kill DDP. Catch any load error and reselect a random sample.
        import random as _rnd
        last = None
        for _ in range(40):
            try:
                return self._getitem_impl(index)
            except Exception as e:
                last = e
                index = _rnd.randint(0, len(self.samples) - 1)
        raise RuntimeError(f"too many unreadable samples; last error: {last}")

    def _getitem_impl(self, index):
        out = super().__getitem__(index)
        index = index % len(self.samples)
        sample = self.samples[index]
        frame_indices = list(range(self.sample_n_frames))

        face_dir = sample.get("face_dir")
        num_persons = int(sample.get("num_persons", 1))
        face_list, masks = [], None
        if face_dir and os.path.isdir(face_dir):
            for p in range(num_persons):
                fn = "src_face.mp4" if p == 0 else f"src_face_{p}.mp4"
                fp = os.path.join(face_dir, fn)
                if os.path.exists(fp):
                    face_list.append(self._load_face_video(fp, frame_indices))
            tj = os.path.join(face_dir, "face_tracks.json")
            if face_list:
                if os.path.exists(tj):
                    # SINGLE and MULTI person both use per-person FACE-REGION masks
                    # (writeup-consistent): each person's expression is injected only
                    # into that person's face tokens; per-frame `present` zeros frames
                    # with no detected face.
                    masks = self._build_masks(json.load(open(tj)), frame_indices)[: len(face_list)]
                else:
                    # no tracks available -> fall back to global (all-ones) so the
                    # face still injects (output * 1 == no mask).
                    h_lat, w_lat = self.height // LATENT_DIV, self.width // LATENT_DIV
                    masks = torch.ones(len(face_list), self.sample_n_frames, h_lat, w_lat)
        if face_list:
            face = torch.stack(face_list, dim=0)               # N,C,F,H,W
            if self.max_persons and face.shape[0] > self.max_persons:
                face = face[: self.max_persons]
                masks = masks[: self.max_persons]
            out["face_pixel_values"] = face
            out["face_masks"] = masks                          # N,F,h_lat,w_lat
        else:
            out["face_pixel_values"] = None
            out["face_masks"] = None
        return out


def face_collate_fn(batch):
    """Stack base fields; zero-pad face_pixel_values / face_masks to the batch max N."""
    out = {}
    keys = ["ref_pixel_values", "first_pixel_values", "third_pixel_values"]
    for k in keys:
        if k in batch[0] and batch[0][k] is not None:
            out[k] = torch.stack([b[k] for b in batch], dim=0)
    out["prompts"] = [b["prompts"] for b in batch]
    out["ref_frame_indices"] = [b.get("ref_frame_indices", []) for b in batch]

    faces = [b.get("face_pixel_values") for b in batch]
    masks = [b.get("face_masks") for b in batch]
    if all(f is None for f in faces):
        out["face_pixel_values"] = None
        out["face_masks"] = None
        return out

    maxN = max((f.shape[0] if f is not None else 0) for f in faces)
    fp_list, mk_list = [], []
    ref_f = next(f for f in faces if f is not None)
    ref_m = next(m for m in masks if m is not None)
    C, F, H, W = ref_f.shape[1:]
    mh, mw = ref_m.shape[2], ref_m.shape[3]
    for f, m in zip(faces, masks):
        if f is None:
            f = torch.zeros(maxN, C, F, H, W)
            m = torch.zeros(maxN, F, mh, mw)
        elif f.shape[0] < maxN:
            pad = maxN - f.shape[0]
            f = torch.cat([f, torch.zeros(pad, C, F, H, W)], dim=0)
            m = torch.cat([m, torch.zeros(pad, F, mh, mw)], dim=0)
        fp_list.append(f)
        mk_list.append(m)
    out["face_pixel_values"] = torch.stack(fp_list, dim=0)  # B,N,C,F,H,W
    out["face_masks"] = torch.stack(mk_list, dim=0)         # B,N,F,h_lat,w_lat
    return out
