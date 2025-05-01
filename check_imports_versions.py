def check_versions():
    print("\n🔍 Checking library versions:\n")

    # NumPy
    try:
        import numpy as np
        print(f"✅ numpy: {np.__version__}")
    except ImportError:
        print("❌ numpy not installed")

    # OpenCV
    try:
        import cv2
        print(f"✅ opencv-python: {cv2.__version__}")
    except ImportError:
        print("❌ opencv-python not installed")

    # Torch
    try:
        import torch
        print(f"✅ torch: {torch.__version__}")
        print(f"   - CUDA available: {torch.cuda.is_available()}")
        print(f"   - CUDA version: {torch.version.cuda}")
        print(f"   - cuDNN version: {torch.backends.cudnn.version()}")
        if torch.cuda.is_available():
            print(f"   - Device count: {torch.cuda.device_count()}")
            print(f"   - Current device: {torch.cuda.current_device()} ({torch.cuda.get_device_name(torch.cuda.current_device())})")
    except ImportError:
        print("❌ torch not installed")

    # TorchVision
    try:
        import torchvision
        print(f"✅ torchvision: {torchvision.__version__}")
    except ImportError:
        print("❌ torchvision not installed")

    # TorchAudio
    # try:
    #     import torchaudio
    #     print(f"✅ torchaudio: {torchaudio.__version__}")
    # except ImportError:
    #     print("❌ torchaudio not installed")

    # # Torch-Scatter
    # try:
    #     import torch_scatter
    #     print(f"✅ torch-scatter: {torch_scatter.__version__}")
    # except ImportError:
    #     print("❌ torch-scatter not installed")

    # PyTorch3D
    try:
        import pytorch3d
        print(f"✅ pytorch3d: {pytorch3d.__version__}")
    except ImportError:
        print("❌ pytorch3d not installed")

if __name__ == "__main__":
    check_versions()