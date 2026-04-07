import torch,bitsandbytes as bnb
print('PyTorch:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
print('GPU:', torch.cuda.get_device_name(0))
print('bitsandbytes:', bnb.__version__)

import bitsandbytes.functional as F
import torch
x = torch.randn(64, 64, device='cuda')
out, state = F.quantize_4bit(x)
print('4-bit quantization: OK')

