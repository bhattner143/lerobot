import os
import cv2
import numpy as np
import torch
import pickle
from pathlib import Path
from model.cloth_model import ClothMeshGATModel

class ReshapeNormalizeImage:
    """
        Resize sample image to (224, 224), normalize to (0, 1)
    """
    def __init__(self, image_size=(224, 224)):
        self.image_size = tuple(image_size)

    def __call__(self, sample):
        sample = np.asarray(cv2.resize(sample, self.image_size).transpose(2, 0, 1) / 255.)
        c, h, w = sample.shape
        return sample

class API_Mesh_GAT:
    def __init__(self, stream_cfg, device):
        # project dir
        self.project_dir = Path('/home/dips/Documents/datasets_lerobot/so100_test/mesh_gat/t_shirt_l3')

        # load checkpoint 
        self.checkpoint_file = os.path.join(self.project_dir, f'checkpoints/{stream_cfg.checkpoint}')
        self.checkpoint = torch.load(self.checkpoint_file, weights_only=False)

        # load template
        self.name_cloth = stream_cfg.name_cloth
        self.template_file = os.path.join(self.project_dir, f'configs/template_{self.name_cloth}.pickle')
        self.template_info = pickle.load(open(self.template_file, mode='rb'))

        # load model
        self.device = device
        self.model = ClothMeshGATModel(self.template_info).to(self.device)

        #Inspect state dict loading
        self._inspect_state_dict_loading(self.model, self.checkpoint['model_state_dict'])
        
        self.model.load_state_dict(self.checkpoint['model_state_dict'])

        self.model.eval()

    def predict(self, input_data):
        # Resize and normalize input data
        transform = ReshapeNormalizeImage()
        input_data = transform(input_data)
        input_data = torch.from_numpy(input_data).float().to(self.device)
        input_data = input_data.unsqueeze(0)

        print(f"input_data shape: {input_data.shape}")
        with torch.no_grad():
            pred_mesh = self.model(input_data)
        return pred_mesh
    

    def _inspect_state_dict_loading(self, model, checkpoint_dict):
        model_keys = set(model.state_dict().keys())
        ckpt_keys = set(checkpoint_dict.keys())

        missing_keys = model_keys - ckpt_keys
        unexpected_keys = ckpt_keys - model_keys

        print(f"\n✅ Model keys: {len(model_keys)}")
        print(f"📦 Checkpoint keys: {len(ckpt_keys)}")
        if missing_keys:
            print(f"❌ Missing keys in checkpoint: {missing_keys}")
        if unexpected_keys:
            print(f"⚠️ Unexpected keys in checkpoint: {unexpected_keys}")
        if not missing_keys and not unexpected_keys:
            print("✅ All keys match!")

if __name__ == "__main__":
    # Example usage
    stream_cfg = type('StreamConfig', (object,), {})()
    stream_cfg.checkpoint = 'finalbestmodel_0299_0.01162.pt'
    stream_cfg.name_cloth = 't_shirt_l3'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    api_mesh_gat = API_Mesh_GAT(stream_cfg, device)
    input_data = np.random.rand(720, 720, 3)  # Dummy input data
    pred_mesh = api_mesh_gat.predict(input_data)
    print(pred_mesh)