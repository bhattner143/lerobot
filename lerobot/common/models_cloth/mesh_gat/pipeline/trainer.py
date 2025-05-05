import os, sys, copy, datetime
import torch, pickle, time, wandb
import torch.distributed as dist
from omegaconf import OmegaConf

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from data.cloth_dataloader import get_dataloader
from model.cloth_model import ClothMeshGATModel
from pipeline.workspace import Workspace
from pipeline.losses import Losses
from utils.visualize_utils import plot_loss_curve

def trainer(cfg):
    # get device for distributed training
    if cfg.distributed:        
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cuda")

    # get training items
    template_info = pickle.load(open(cfg.template_dir, mode='rb'))
    train_loader, val_loader = get_dataloader(cfg)
    model = ClothMeshGATModel(template_info).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    cal_loss = Losses(cfg, device, template_info).to(device)

    if cfg.distributed:
        dummy_input = torch.randn(cfg.batch_size, 3, 244, 244).to(device)
        _ = model(dummy_input)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])

    # initialize workspace
    workspace = Workspace(model,opt,cal_loss,template_info,cfg,device)

    #locate save experiment dir
    cur_day = datetime.datetime.now().strftime('%Y-%m-%d')
    day_dir = os.path.join(cfg.checkpoint_dir, cur_day)
    os.makedirs(day_dir, exist_ok=True)
    cur_time = datetime.datetime.now().strftime('%H-%M-%S')
    task_save_dir = os.path.join(day_dir, cur_time)
    os.makedirs(task_save_dir, exist_ok=True)

    # save current cfg as YAML
    cfg_save_path = os.path.join(task_save_dir, 'config.yaml')
    OmegaConf.save(cfg, cfg_save_path)
    
    # save model initialize
    name_log = None
    best_train_loss = float('inf')
    best_train_epoch = -1
    best_train_model = {}
    loss_record = {'train': [], 'val': []}
    loss_plot_path = os.path.join(task_save_dir, f"loss_curve.png")
    t_init = time.time()

    # Initialize wandb for logging
    if not cfg.distributed or (cfg.distributed and dist.get_rank() == 0):
        wandb.init(project="MeshGAT", name=f"{cur_day}-{cur_time}")
        wandb.config.update(OmegaConf.to_container(cfg, resolve=True))

    # forward mode for epoch_size
    for ne in range(cfg.epoch_size):
        # train model with train data_loader
        workspace.model.train()
        train_loss, lr = workspace.forward('train', train_loader, epoch=ne)
        loss_record['train'].append(train_loss)
        
        # validate model with val data_loader
        with torch.no_grad():
            workspace.model.eval()
            val_loss, _ = workspace.forward('val', val_loader, epoch=ne)
            loss_record['val'].append(val_loss)
            
            # update best train epoch
            if val_loss < best_train_loss:
                # remove the previous bestmodel
                if name_log:
                    os.remove(name_log)
                
                # copy the curernt best model
                if not cfg.distributed or (cfg.distributed and dist.get_rank() == 0):
                    best_train_epoch = ne
                    best_train_loss = val_loss
                    if not cfg.distributed:
                        best_train_model = copy.deepcopy(workspace.model.state_dict())
                    else:
                        best_train_model = copy.deepcopy(workspace.model.module.state_dict())

                    # save the current best mdoel
                    name_log = os.path.join(task_save_dir, f"bestmodel_{best_train_epoch:04d}_{best_train_loss:.5f}.pt")
                    torch.save({'epoch': ne,
                                'model_state_dict': best_train_model}, name_log)

        # log current states    
        if not cfg.distributed or (cfg.distributed and dist.get_rank() == 0):
            wandb.log({"Train Loss": train_loss, "Learning Rate": lr, "Validation Loss": val_loss})

        # save model with save step
        if ne % cfg.save_step == 0:
            if not cfg.distributed or (cfg.distributed and dist.get_rank() == 0):
                torch.save({'epoch': ne, 
                            'model_state_dict': best_train_model, 
                            'optimizer_state_dict': workspace.opt.state_dict()},
                            os.path.join(task_save_dir, f"model_{ne:04d}_{val_loss:.5f}.pt"))
                
                # plot and save the loss figure
                if os.path.exists(loss_plot_path):
                    os.remove(loss_plot_path)
                plot_loss_curve(loss_record, loss_plot_path)
            
    # save best model
    if not cfg.distributed or (cfg.distributed and dist.get_rank() == 0):
        torch.save({'model_state_dict': best_train_model},
                    os.path.join(task_save_dir, f"finalbestmodel_{best_train_epoch:04d}_{best_train_loss:.5f}.pt"))
        
        print(f"It took {int(time.time() - t_init)} s for training and validat {cfg.epoch_size} Epochs.")

