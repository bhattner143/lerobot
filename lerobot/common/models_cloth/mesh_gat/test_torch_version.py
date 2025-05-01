import torch
import torchvision
import torchaudio

try:
    import pytorch3d
    pytorch3d_version = pytorch3d.__version__
except ImportError:
    pytorch3d_version = "PyTorch3D is not installed"

try:
    import torch_scatter
    torch_scatter_version = torch_scatter.__version__
except ImportError:
    torch_scatter_version = "torch-scatter is not installed"

print("✅ PyTorch Version:", torch.__version__)
print("🎨 TorchVision Version:", torchvision.__version__)
print("🔊 TorchAudio Version:", torchaudio.__version__)
print("🎭 PyTorch3D Version:", pytorch3d_version)
print("🔄 Torch-Scatter Version:", torch_scatter_version)
print("🚀 CUDA Available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("🧠 CUDA Device Count:", torch.cuda.device_count())
    print("📍 Current CUDA Device:", torch.cuda.current_device())
    print("🎮 CUDA Device Name:",  torch.cuda.get_device_name(torch.cuda.current_device()))