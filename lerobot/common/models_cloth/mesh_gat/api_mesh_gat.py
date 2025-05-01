import os
import cv2
import numpy as np
import torch
import pickle
from mesh_gat.model.cloth_model import ClothModel

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
        self.project_dir = os.path.dirname(os.path.abspath(__file__))

        # load checkpoint 
        self.checkpoint_file = os.path.join(self.project_dir, f'checkpoints/{stream_cfg.checkpoint}')
        self.checkpoint = torch.load(self.checkpoint_file, weights_only=False)

        # load template
        self.name_cloth = stream_cfg.name_cloth
        self.template_file = os.path.join(self.project_dir, f'configs/template_{self.name_cloth}.pickle')
        self.template_info = pickle.load(open(self.template_file, mode='rb'))

        # load model
        self.device = device
        self.model = ClothModel(self.template_info).to(self.device)
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
