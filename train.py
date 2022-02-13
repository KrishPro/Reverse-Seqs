"""
Written by KrishPro @ KP

This file'll contain the training loop
"""

from tqdm import tqdm
import time
from torch.utils.tensorboard import SummaryWriter
from typing import Generator
import torch.optim as optim
import torch.nn as nn
import torch
from data import create_batch

from model import Transformer

VOCAB_SIZE = 13
SEQ_LEN = 10
D_MODEL = 512
NHEAD = 8
LEARNING_RATE = 1e-4
DROPOUT = 0.2
FFN_HID_DIM = 512 * 4
BATCH_SIZE = 32
NUM_ENCODER_LAYERS = 6
NUM_DECODER_LAYERS = 6

def create_optimizer(params: Generator, learning_rate: float) -> tuple[optim.Adam, nn.CrossEntropyLoss]:
    optimizer = optim.Adam(params, lr=learning_rate, betas=(0.9, 0.98), eps=1e-9)
    criterion = nn.CrossEntropyLoss()
    return optimizer, criterion

def setup_tensorboard():
    writer = SummaryWriter(f"runs/{time.time()}")
    return writer, 0

def train_step(transformer: Transformer, optimizer: optim.Adam, criterion: nn.CrossEntropyLoss, device="cpu"):
    src, tar = create_batch(BATCH_SIZE, SEQ_LEN, device)
    T, N = tar.shape

    tar_input = tar[:-1]
    assert tar_input.shape == (T-1, N)
    model_output: torch.Tensor = transformer(src, tar_input)
    assert model_output.shape == (T-1, N, VOCAB_SIZE)

    model_output = model_output.view((T-1)*N, VOCAB_SIZE)
    tar_output = tar[1:].reshape((T-1)*N)

    loss: torch.Tensor = criterion(model_output, tar_output)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

transformer = Transformer(D_MODEL, VOCAB_SIZE, NHEAD, NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, FFN_HID_DIM, DROPOUT).to(device)
optimizer, criterion = create_optimizer(transformer.parameters(), LEARNING_RATE)

writer, global_step = setup_tensorboard()

def save_checkpoint(transformer: Transformer):
    torch.save(transformer.state_dict(), "checkpoints/latest.pth")

    i = 0
    while True:
        loss = train_step(transformer, optimizer, criterion, device=device)
        writer.add_scalar("loss", loss, global_step=global_step)
        global_step += 1
    if i % 500 == 0:
        save_checkpoint(transformer)

        i += 1
 
