import cv2
import numpy as np
import torch
from typing import Tuple
from raft_extractor.utils._types import *


def analyze_video(video: str, keep: str = "first") -> np.ndarray:
    cap = cv2.VideoCapture(video)
    try:
        # frame per second for the current video in order to average the frames
        _FPS = int(cap.get(cv2.CAP_PROP_FPS))
        fps = _FPS + 1
    except ValueError:
        assert f"Cannot convert video {video} fps to integer"
    # print(f'Proccessing {video} with: {fps} fps')
    success = True
    batches = []
    count = 0

    while success:
        success, frame = cap.read()
        if success:
            # frame = cv2.resize(frame, (124, 124))
            tmp_frame = torch.from_numpy(np.moveaxis(frame, -1, 0))
            if keep == EXPORT_FIRST:
                if count % _FPS == 0:
                    batches.append(tmp_frame)
            if keep == EXPORT_LAST:
                if count % _FPS == _FPS - 1:
                    batches.append(tmp_frame)
            if keep == EXPORT_ALL:
                batches.append(tmp_frame)
        count += 1

    return torch.stack(batches)


def partion_frames(frames:torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    odd_partition = frames[::2, :] 
    even_partition = frames[1::2, :]
    if odd_partition.shape[0] > even_partition.shape[0]:
        frame = even_partition[None, -1]
        even_partition = torch.cat((even_partition, frame))
    return odd_partition, even_partition
