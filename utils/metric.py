#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.

import os
import time
import torch
import functools
import numpy as np
from collections import defaultdict, deque

__all__ = [
    "AverageMeter",
    "MeterBuffer",
    "get_total_and_free_memory_in_Mb",
    "occupy_mem",
    "gpu_mem_usage",
]


def get_total_and_free_memory_in_Mb(cuda_device):
    devices_info_str = os.popen(
        "nvidia-smi --query-gpu=memory.total,memory.used --format=csv,nounits,noheader"
    )
    devices_info = devices_info_str.read().strip().split("\n")
    total, used = devices_info[int(cuda_device)].split(",")
    return int(total), int(used)


def occupy_mem(cuda_device, mem_ratio=0.95):
    """
    pre-allocate gpu memory for training to avoid memory Fragmentation.
    """
    total, used = get_total_and_free_memory_in_Mb(cuda_device)
    max_mem = int(total * mem_ratio)
    block_mem = max_mem - used
    x = torch.cuda.FloatTensor(256, 1024, block_mem)
    del x
    time.sleep(5)


def gpu_mem_usage(device=None):
    """
    Compute the GPU memory usage for the current device (MB).
    """
    if device is None:
        device = torch.cuda.current_device()
    mem_usage_bytes = torch.cuda.memory_allocated(device)
    return mem_usage_bytes / (1024 * 1024)


class AverageMeter:
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=50):
        self._deque = deque(maxlen=window_size)
        self._total = 0.0
        self._count = 0

    def update(self, value):
        self._deque.append(value)
        self._count += 1
        self._total += value

    @property
    def median(self):
        d = np.array(list(self._deque))
        return np.median(d)

    @property
    def avg(self):
        # if deque is empty, nan will be returned.
        d = np.array(list(self._deque))
        return d.mean()

    @property
    def global_avg(self):
        return self._total / max(self._count, 1e-5)

    @property
    def latest(self):
        return self._deque[-1] if len(self._deque) > 0 else None

    @property
    def total(self):
        return self._total

    def reset(self):
        self._deque.clear()
        self._total = 0.0
        self._count = 0

    def clear(self):
        self._deque.clear()


class MeterBuffer(defaultdict):
    def __init__(self, window_size=20):
        factory = functools.partial(AverageMeter, window_size=window_size)
        super().__init__(factory)
        self._artifacts = {}

    @staticmethod
    def _is_scalar(value):
        if isinstance(value, (int, float)):
            return True
        if isinstance(value, torch.Tensor):
            return value.numel() == 1
        if isinstance(value, np.ndarray):
            return value.size == 1
        return False

    def update(self, values=None, **kwargs):
        if values is None:
            values = {}
        values.update(kwargs)
        for k, v in values.items():
            if self._is_scalar(v):
                if isinstance(v, torch.Tensor):
                    v = v.detach()
                self[k].update(v)
            else:
                self._artifacts[k] = v

    def get_artifact(self, key, default=None):
        return self._artifacts.get(key, default)

    def clear_artifacts(self):
        self._artifacts.clear()

    def reset(self):
        """清空所有 meters 和 artifacts（删除所有 keys）"""
        super().clear()
        self.clear_artifacts()

    def clear_meters(self):
        """清空所有 scalar meters 的历史记录（保留 key）"""
        for meter in self.values():
            meter.clear()

    def get_filtered_meter(self, filter_key="time"):
        return {k: v for k, v in self.items() if filter_key in k}