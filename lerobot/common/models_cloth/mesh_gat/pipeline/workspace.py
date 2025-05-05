import os, sys, time
import torch, wandb
from pytorch3d.loss import chamfer_distance
import torch.distributed as dist
from torchvision.transforms import Compose
from torch.utils.data import DataLoader

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from model.cloth_model import ClothMeshGATModel
from data.cloth_dataloader import get_dataloader
from utils.configs_utils import get_args
from utils.visualize_utils import *
from utils.render_utils import *
from utils.train_utils import *
from utils.cloth_utils import *

class Workspace:
    def __init__(self,
                 model,
                 optimizer,
                 cal_loss,
                 template_info,
                 cfg,
                 device,
                 dtype=torch.float32,
                 ):

        self.model = model
        self.opt = optimizer
        self.cal_loss = cal_loss
        self.template_info = template_info

        self.data_dir = cfg.dataset_dir

        self.batch_size = cfg.batch_size
        self.epoch_size = cfg.epoch_size
        self.image_size = cfg.image_size
        self.lr = cfg.lr
        self.cur_lr = self.lr
        self.save_step = cfg.save_step
        self.schedule_step = cfg.schedule_step

        self.dtype = dtype
        self.device = device

        # self.L1Loss = torch.nn.L1Loss().to(self.device)
        # self.renderer = Renderer(image_size=self.image_size, dtype=dtype, device=self.device)

        self.log_train_losses = AverageMeter()
        self.log_compare_losses = AverageMeter()

    # send batch data to device
    def send_to_device(self, batch):
        for k, v in batch.items():
            if k == 'image_name': continue
            batch[k] = v.to(self.device)

    # ------------------------------------------------------
    # ---------------------- Losses ------------------------
    # ------------------------------------------------------
    # def vertex_loss(self, pred_mesh, true_mesh):
    #     '''
    #     calculate mean vertex loss of each batch
    #     output: batch_vertex_losses (Batch_size x Loss_scaler)
    #     '''
    #     # calculate batch vertex losses
    #     batch_vertex_losses = [self.L1Loss(pred_mesh[nb], true_mesh[nb]) for nb in range(true_mesh.shape[0])]
        
    #     return batch_vertex_losses
    
    # # keypoint loss between pred_mesh and true_mesh
    # def keypoint_loss(self, pred_mesh, true_mesh):
    #     '''
    #     calculate keypoint loss of each batch
    #     output: batch_keypoint_losses (Batch_size x Loss_scaler)
    #     '''
    #     # get keypoint idx from template
    #     keypoint_idx = self.template_info['keypoint_idx']
        
    #     # calculate batch keypoint losses
    #     batch_keypoint_losses = [self.L1Loss(pred_mesh[nb, keypoint_idx], true_mesh[nb, keypoint_idx]) for nb in range(true_mesh.shape[0])]
        
    #     return batch_keypoint_losses

    # # vertex wise losses between pred vertices and true vertices
    # def vertex_wise_losses(self, pred_mesh, true_mesh, loss_dict):
    #     # assert pred and true batch size
    #     assert pred_mesh.shape[0] == true_mesh.shape[0]
        
    #     # calculate vertex_loss of each batch
    #     batch_vertex_losses = torch.stack(self.vertex_loss(pred_mesh, true_mesh))
        
    #     # update loss_dict with mean batch_losses
    #     if 'vertex_loss' in loss_dict: 
    #         loss_dict['vertex_loss'] += torch.mean(batch_vertex_losses)
    #     else: 
    #         loss_dict['vertex_loss'] = torch.mean(batch_vertex_losses)

    #     # calculate keypoint loss of each batch
    #     batch_keypoint_losses = torch.stack(self.keypoint_loss(pred_mesh, true_mesh))
        
    #     # update loss_dict with mean batch_losses
    #     if 'keypoint_loss' in loss_dict: 
    #         loss_dict['keypoint_loss'] += torch.mean(batch_keypoint_losses)
    #     else: 
    #         loss_dict['keypoint_loss'] = torch.mean(batch_keypoint_losses)

    #     # return batch_losses
    #     return loss_dict

    # # weight pose alignment losses
    # def weighted_losses(self, losses):
    #     # get pose align loss weight dict
    #     weight = {'vertex_loss': lambda cst: 10. ** 0 * cst,
    #               'corner_loss': lambda cst: 10. ** -1 * cst,
    #               'keypoint_loss': lambda cst: 10. ** -1 * cst,
    #               'depth_loss': lambda cst: 10. ** 0 * cst,
    #               'silhouette_loss': lambda cst: 10. ** 0 * cst,
    #               'chamfer_loss': lambda cst: 10. ** 0 * cst,
    #               }

    #     # init weight_loss with zero
    #     weight_losses = torch.tensor([0.]).to(self.device)
    #     # weight all loss
    #     for l in losses:
    #         weight_losses += weight[l](losses[l])
    #     return weight_losses

    # ------------------------------------------------------
    # -------------------- Scheduler -----------------------
    # ------------------------------------------------------
    def adjust_learning_rate(self, opt, epoch):
        self.cur_lr = self.lr * (0.1 ** (epoch // self.schedule_step))
        for param_group in opt.param_groups:
            param_group['lr'] = self.cur_lr

    # ------------------------------------------------------
    # -------------------- Processes -----------------------
    # ------------------------------------------------------
    def forward(self, phase, data_loader, epoch=None):
        # init model, optimizer, phase, epoch, is_training
        model = self.model
        opt = self.opt
        is_training = True if phase == 'train' else False
        is_testing = True if phase == 'test' else False
        t_0 = time.time()

        # loop over all batches
        for it, batch in enumerate(data_loader):
            if (dist.is_initialized() and dist.get_rank() == 0) or not dist.is_initialized():
                world_size = get_world_size()
                if phase == 'train':
                    show_process(it, 
                                len(data_loader), 
                                prefix='{}/Epoch:{}/Finish:{}/Entire:{}/Used:{:.1f}s/Loss:{:.3f}/Compare:{:.3f}'.format(phase, 
                                            epoch, 
                                            (it+1)*self.batch_size*world_size, 
                                            len(data_loader)*self.batch_size*world_size, 
                                            time.time()-t_0,
                                            self.log_train_losses.avg,
                                            self.log_compare_losses.avg))
                elif phase == 'val':
                    show_process(it, 
                                len(data_loader), 
                                prefix='{}/Epoch:{}/Finish:{}/Entire:{}/Used:{:.1f}s/Loss:{:.3f}/Compare:{:.3f}'.format(phase, 
                                            epoch, 
                                            (it+1)*self.batch_size, 
                                            len(data_loader)*self.batch_size, 
                                            time.time()-t_0,
                                            self.log_train_losses.avg,
                                            self.log_compare_losses.avg))

            # initialize for each iteration
            opt.zero_grad()
            self.send_to_device(batch)

            # pred vertices form image_simu
            pred_meshes = model(batch['image'])

            # calculate losses
            total_loss, losses, compare_loss = self.cal_loss(pred_meshes, batch['mesh'], epoch)
            # loss_dict = {}
            # loss_dict = self.vertex_wise_losses(pred_meshes, batch['mesh'], loss_dict)
            # loss = self.weighted_losses(loss_dict)
            batch_size = batch['image'].shape[0]
            self.log_train_losses.update(total_loss.item(), batch_size)
            self.log_compare_losses.update(compare_loss.item(), batch_size)
            
            # optimize forward
            if is_training:
                # backward loss
                # if using distributed the loss will synchronized automatically
                total_loss.backward()
                # walk along the gradient
                opt.step()
                # adjust learning rate
                self.adjust_learning_rate(opt, epoch)
                
        # return averaged train losses
        return self.log_train_losses.avg, self.cur_lr
    
    def forward_and_visualize_prediction(self, phase, data_loader, epoch=None):
        # init model, optimizer, phase, epoch, is_training
        model = self.model
        opt = self.opt
        is_training = True if phase == 'train' else False
        is_testing = True if phase == 'test' else False
        is_testing_real = True if phase == 'test_real' else False
        log_predict = []  # List to store first item from each batch

        # loop over all batches
        for it, batch in enumerate(data_loader):
            show_process(it, 
                         len(data_loader), 
                         prefix='{}/{}/{}/{}/Loss:{:.3f}'.format(phase, 
                                    epoch, 
                                    (it+1)*self.batch_size, 
                                    len(data_loader)*self.batch_size, 
                                    self.log_train_losses.avg))

            # initialize for each iteration
            opt.zero_grad()
            self.send_to_device(batch)

            # pred vertices form image_simu
            pred_meshes = model(batch['image'])

            # calculate losses
            if is_training or is_testing:
                # loss_dict = {}
                # loss_dict = self.vertex_wise_losses(pred_meshes, batch['mesh'], loss_dict)
                # loss = self.weighted_losses(loss_dict)
                total_loss, losses, compare_loss = self.cal_loss(pred_meshes, batch['mesh'], epoch)
                
                batch_size = batch['image'].shape[0]
                self.log_train_losses.update(total_loss.item(), batch_size)
            
            # optimize forward
            if is_training:
                # backward loss and optimize step
                total_loss.backward()
                opt.step()

                # adjust learning rate
                self.adjust_learning_rate(opt, epoch)

            # store the first item of each batch
            if is_testing_real:
                log_predict.append({
                'name': batch['image_name'][0], # name of this item
                'image_input': batch['image_input'][0].cpu(),  # First image in batch
                'image_transform': batch['image'][0].cpu(),
                'predict_mesh': pred_meshes[0].cpu()  # First predicted mesh
            })
            else:
                log_predict.append({
                    'name': batch['image_name'][0], # name of this item
                    'image_input': batch['image_input'][0].cpu(),  # First image in batch
                    'image_transform': batch['image'][0].cpu(),
                    'true_mesh': batch['mesh'][0].cpu(),  # First ground truth mesh
                    'predict_mesh': pred_meshes[0].cpu()  # First predicted mesh
                })

        # return averaged train losses
        if is_training or is_testing:
            return self.log_train_losses.avg, log_predict
        elif is_testing_real:
            return log_predict

