
import torch
from channel import ZadoffChu, Normalize

M=64
    
pilot = ZadoffChu(order=1, length=M)
pilot, _ = Normalize(pilot.unsqueeze(0))
torch.save(pilot, 'pilot.pt')