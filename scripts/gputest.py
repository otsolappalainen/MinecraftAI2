import numpy as np
import torch

print("Numpy version:", np.__version__)
print("Torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())