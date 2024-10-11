# TODO: Adapted from cli
from typing import Callable, List, Optional

import numpy as np


def compute_num_context(init_video_length, context_size, context_overlap):
    step = context_size - context_overlap
    num_windows = (init_video_length - context_size) // step + 1
    return num_windows


def compute_context_indices(num_context, context_size, context_overlap):
    indices = []
    for i in range(num_context):
        start_index = i * (context_size - context_overlap)
        end_index = start_index + context_size - 1
        indices.append((start_index, end_index))
    return indices


def ordered_halving(val):
    bin_str = f"{val:064b}"
    bin_flip = bin_str[::-1]
    as_int = int(bin_flip, 2)

    return as_int / (1 << 64)


def uniform(
    step: int = ...,
    num_frames: int = ...,
    context_size: Optional[int] = None,
    context_stride: int = 3,
    context_overlap: int = 4,
    closed_loop: bool = True,
):
    if num_frames <= context_size:
        yield list(range(num_frames))
        return

    context_stride = min(
        context_stride, int(np.ceil(np.log2(num_frames / context_size))) + 1
    )

    for context_step in 1 << np.arange(context_stride):
        pad = int(round(num_frames * ordered_halving(step)))
        for j in range(
            int(ordered_halving(step) * context_step) + pad,
            num_frames + pad + (0 if closed_loop else -context_overlap),
            (context_size * context_step - context_overlap),
        ):
            next_itr = []
            for e in range(j, j + context_size * context_step, context_step):
                if e >= num_frames:
                    e = num_frames - 2 - e % num_frames
                next_itr.append(e)

            yield next_itr


def get_context_scheduler(name: str) -> Callable:
    if name == "uniform":
        return uniform
    else:
        raise ValueError(f"Unknown context_overlap policy {name}")


def get_total_steps(
    scheduler,
    timesteps: List[int],
    num_steps: Optional[int] = None,
    num_frames: int = ...,
    context_size: Optional[int] = None,
    context_stride: int = 3,
    context_overlap: int = 4,
    closed_loop: bool = True,
):
    return sum(
        len(
            list(
                scheduler(
                    i,
                    num_steps,
                    num_frames,
                    context_size,
                    context_stride,
                    context_overlap,
                )
            )
        )
        for i in range(len(timesteps))
    )
