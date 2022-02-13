"""
Written by KrishPro @ KP
"""

from model import Transformer
from train import BATCH_SIZE, D_MODEL, SEQ_LEN, VOCAB_SIZE, NHEAD, NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, FFN_HID_DIM, DROPOUT
import torch.nn.functional as F
from data import SOS_TOKEN, EOS_TOKEN, create_batch
import torch

def take_input_src(default: int = 498375):
    print(f"enter a sequence on numbers, like '{default}'")
    print("ProTip: leave it blank to use the default seq")
    src = str(input(f"({default}) => ").strip())
    try:
        src = list(map(int, str(int(src))))
        assert len(src) > 1, "Seq Len should be greater than 1"
        print(f"Input Sequence: {src}")
    except Exception as e:
        src = list(map(int, str(int(f'{default}'))))
        print(f"fatal: {e}")
        print(f"Using default seq: {src}")

    return src

def append_tokens(src: list[int]):
    return [SOS_TOKEN] + src + [EOS_TOKEN]

def convert_to_tensor(src: list[int], device=None):
    src = torch.tensor(src, device=device).unsqueeze(1)
    return src

@torch.no_grad()
def reverse(src: torch.Tensor, tar: torch.Tensor, transformer: Transformer):
    T, N = tar.shape
    tar_output: torch.Tensor = transformer(src, tar[:-1])
    VOCAB_SIZE = tar_output.size(2)
    loss = F.cross_entropy(tar_output.view((T-1)*N, VOCAB_SIZE), tar[1:].reshape((T-1)*N))
    return tar_output.argmax(-1), loss

def load_transformer(checkpoint_path: str = "checkpoints/latest.pth", device: torch.device = "cuda"):
    transformer = Transformer(D_MODEL, VOCAB_SIZE, NHEAD, NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, FFN_HID_DIM, DROPOUT).to(device)

    transformer.load_state_dict(torch.load(checkpoint_path))

    return transformer

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transformer = load_transformer(device=device)

    src, tar = create_batch(BATCH_SIZE, SEQ_LEN, device)


    tar_out, loss = reverse(src, tar, transformer)

    for i in range(min(src.size(1), 10)):
        print(f"=> {src.T[i][1:-1]}")
        print(f"=> {torch.flip(tar_out.T[i][:-1], dims=(0,))}")
        print(f"=> {tar_out.T[i][:-1]}")
        print("")

    print(f"Average Loss: {loss:.25f}")

if __name__ == '__main__':
    main()