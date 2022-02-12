"""
Written by KrishPro @ KP

This file'll contain all code releated data management
"""

import random
import torch

SOS_TOKEN = 11
EOS_TOKEN = 12

def create_batch(batch_size: int, seq_len: int = 10, device: torch.device = torch.device("cpu")):
    """
    Generates a data sample

    First. we get numbers from 0 to 9, then we shuffled them and stored as many as batch_size
    Second. we loop over src and just reversed the tensors
    Laslty. we concatenated both src & tar at dim=0, 
    then transposed them to get seq_len before batch_size in shape
    """

    assert seq_len <= 10, "Seq Len greater than 10 is not supported"

    src, tar = [], []
    for _ in range(batch_size):
        src.append([SOS_TOKEN] + random.sample(list(range(seq_len)), seq_len-2) + [EOS_TOKEN])
        tar.append([SOS_TOKEN] + list(reversed(src[-1][1:-1])) + [EOS_TOKEN])

    src = torch.tensor(src, device=device)
    tar = torch.tensor(tar, device=device)

    assert tuple(src.shape) == tuple(tar.shape) == (batch_size, seq_len), f"{tuple(src.shape)=}, {tuple(tar.shape)=}"

    return src.T, tar.T
