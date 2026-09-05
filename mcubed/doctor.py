import json
import platform
import torch
from .common import device_for

if __name__ == '__main__':
    device = device_for('auto')
    x = torch.ones(8, device=device, requires_grad=True)
    (x.square().sum()).backward()
    print(json.dumps({'python': platform.python_version(), 'torch': str(torch.__version__),
        'machine': platform.machine(), 'mps': torch.backends.mps.is_available(),
        'cuda': torch.cuda.is_available(), 'selected': device,
        'backward_ok': bool(torch.all(x.grad == 2).item())}, indent=2))
