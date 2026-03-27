"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""
from typing import List, Dict, Tuple, Optional
from torch.utils.data import Dataset
import torch.nn.functional as F
import numpy as np
import torch

class BaseDataset(Dataset):
    def __init__(
        self, 
        data_path: str, 
        device: str = 'cpu'
    ):
        
        self.data_path = data_path
        self.device = device