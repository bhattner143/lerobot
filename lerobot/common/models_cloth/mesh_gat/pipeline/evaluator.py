import os, sys
import torch
import pickle
import datetime
import numpy as np
from omegaconf import OmegaConf

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from data.cloth_dataloader import get_test_dataloader, get_test_real_dataloader
from model.cloth_model import ClothMeshGATModel
from pipeline.workspace import Workspace
from pipeline.losses import Losses
from utils.visualize_utils import plot_prediction_result_with_label, plot_prediction_result_without_true_label

def evaluator(cfg):
    # get training items
    device = torch.device('cuda')
    template_info = pickle.load(open(cfg.template_dir, mode='rb'))
    test_loader = get_test_dataloader(cfg)
    model = ClothMeshGATModel(template_info).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    cal_loss = Losses(cfg, device, template_info)
    workspace = Workspace(model,opt,cal_loss, template_info,cfg, device)

    #locate save experiment dir
    cur_day = datetime.datetime.now().strftime('%Y-%m-%d')
    day_dir = os.path.join(cfg.predict_dir, cur_day)
    os.makedirs(day_dir, exist_ok=True)
    cur_time = datetime.datetime.now().strftime('%H-%M-%S')
    task_save_dir = os.path.join(day_dir, cur_time)
    os.makedirs(task_save_dir, exist_ok=True)

    # save current cfg as YAML
    cfg_save_path = os.path.join(task_save_dir, 'config.yaml')
    OmegaConf.save(cfg, cfg_save_path)
    
    # load checkpoint to model
    print(cfg.checkpoint_file)
    checkpoint = torch.load(cfg.checkpoint_file, weights_only=False)
    print("load success")
    workspace.model.load_state_dict(checkpoint['model_state_dict'])

    # test model with test data loader
    with torch.no_grad():
        workspace.model.eval()
        if cfg.store_pred:
            test_loss, results = workspace.forward_and_visualize_prediction('test', test_loader)
        else:
            test_loss = workspace.forward('test', test_loader, store_pred=True)

    print(f"Test finish with average loss: {test_loss}")
    
    # Convert each tensor in the list of dictionaries to a NumPy array
    for item in results:
        name = item['name']

        img_origin = item['image_input'].cpu().numpy()
        # img_origin = np.transpose(img_origin, (1, 2, 0))
        img_origin = img_origin.astype(np.uint8)

        img_trans = item['image_transform'].cpu().numpy()
        img_trans = np.transpose(img_trans, (1, 2, 0))
        img_trans = (img_trans * 255).astype(np.uint8) 

        mesh_true = item['true_mesh'].cpu().numpy()
        mesh_pred = item['predict_mesh'].cpu().numpy()

        predict_save_path = os.path.join(task_save_dir, f'{name}.predicted_result.png')
        plot_prediction_result_with_label(name, img_origin, img_trans, mesh_true, mesh_pred, predict_save_path)

def evaluator_real(cfg):
    # get training items
    device = torch.device('cuda')
    template_info = pickle.load(open(cfg.template_dir, mode='rb'))
    test_loader = get_test_real_dataloader(cfg)
    model = ClothMeshGATModel(template_info).to(device)
    cal_loss = Losses(cfg, device, template_info)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    workspace = Workspace(model,opt,cal_loss,template_info,cfg, device)

    #locate save experiment dir
    cur_day = datetime.datetime.now().strftime('%Y-%m-%d')
    day_dir = os.path.join(cfg.predict_dir, cur_day)
    os.makedirs(day_dir, exist_ok=True)
    cur_time = datetime.datetime.now().strftime('%H-%M-%S')
    task_save_dir = os.path.join(day_dir, cur_time)
    os.makedirs(task_save_dir, exist_ok=True)

    # save current cfg as YAML
    cfg_save_path = os.path.join(task_save_dir, 'config.yaml')
    OmegaConf.save(cfg, cfg_save_path)
    
    # load checkpoint to model
    checkpoint = torch.load(cfg.checkpoint_file, weights_only=False)
    workspace.model.load_state_dict(checkpoint['model_state_dict'])

    # test model with test data loader
    with torch.no_grad():
        workspace.model.eval()
        results = workspace.forward_and_visualize_prediction('test_real', test_loader)
    
    # Convert each tensor in the list of dictionaries to a NumPy array
    for item in results:
        name = item['name']

        img_origin = item['image_input'].cpu().numpy()
        img_origin = img_origin.astype(np.uint8)

        img_trans = item['image_transform'].cpu().numpy()
        img_trans = np.transpose(img_trans, (1, 2, 0))
        img_trans = (img_trans * 255).astype(np.uint8) 

        mesh_pred = item['predict_mesh'].cpu().numpy()

        predict_save_path = os.path.join(task_save_dir, f'{name}.predicted_result_from_real.png')
        plot_prediction_result_without_true_label(name, img_origin, img_trans, mesh_pred, predict_save_path)

    print("Finish prediction from real dataset.")
