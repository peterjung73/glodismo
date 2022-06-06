import torch
device = 'cpu'

if torch.cuda.is_available():
    device = 'cuda'
    print("Using CUDA")
