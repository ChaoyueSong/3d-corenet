import os
import torch

from models.networks.sync_batchnorm import DataParallelWithCallback
from models.ver2ver_model import Ver2VerModel

class Ver2VerTrainer():
    """
    Trainer creates the model and optimizers, and uses them to
    updates the weights of the network while reporting losses
    and the latest visuals to visualize the progress in training.
    """

    def __init__(self, opt):
        self.opt = opt
        self.ver2ver_model = Ver2VerModel(opt)
        if len(opt.gpu_ids) > 1:
            self.ver2ver_model = DataParallelWithCallback(self.ver2ver_model,
                                                          device_ids=opt.gpu_ids)
            self.ver2ver_model_on_one_gpu = self.ver2ver_model.module
        else:
            self.ver2ver_model.to(opt.gpu_ids[0])
            self.ver2ver_model_on_one_gpu = self.ver2ver_model

        if opt.isTrain:
            self.optimizer = self.ver2ver_model_on_one_gpu.create_optimizers(opt)
            self.old_lr = opt.lr
            if opt.continue_train and opt.which_epoch == 'latest':
                checkpoint = torch.load(os.path.join(opt.checkpoints_dir, opt.dataset_mode, 'optimizer.pth'))
                self.optimizer.load_state_dict(checkpoint['G'])
                self.old_lr = checkpoint['lr']

    def train_model(self, identity_points, pose_points, gt_points, id_face):
        self.optimizer.zero_grad()
        losses, out = self.ver2ver_model(identity_points, pose_points, gt_points, id_face, mode='train')
        loss = sum(losses.values()).mean()
        loss.backward()
        self.optimizer.step()
        self.losses = losses
        self.out = out

    def get_latest_losses(self):
        return {**self.losses}

    def get_latest_generated(self):
        return self.out['fake_points']

    def update_learning_rate(self, epoch):
        self.update_learning_rate(epoch)

    def save(self, epoch):
        self.ver2ver_model_on_one_gpu.save(epoch)
        if epoch == 'latest':
            torch.save({'G': self.optimizer.state_dict(),
                        'lr':  self.old_lr,
                        }, os.path.join(self.opt.checkpoints_dir, self.opt.dataset_mode, 'optimizer.pth'))

    ##################################################################
    # Helper functions
    ##################################################################

    def update_learning_rate(self, epoch):
        if epoch > self.opt.niter:
            lrd = self.opt.lr / self.opt.niter_decay
            new_lr = self.old_lr - lrd
        else:
            new_lr = self.old_lr

        if new_lr != self.old_lr:
            new_lr_G = new_lr

            for param_group in self.optimizer.param_groups:
                param_group['lr'] = new_lr_G
            print('update learning rate: %f -> %f' % (self.old_lr, new_lr))
            self.old_lr = new_lr
