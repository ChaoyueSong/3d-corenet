import os
import sys
import pymesh
import torch

from data.human_data import SMPL_DATA
from data.animal_data import SMAL_DATA
from ver2ver_trainer import Ver2VerTrainer
from options.train_options import TrainOptions
from util.iter_counter import IterationCounter
from util.util import print_current_errors


# parse options
opt = TrainOptions().parse()

# print options to help debugging
print(' '.join(sys.argv))

# load the dataset
if opt.dataset_mode == 'human':
    dataset = SMPL_DATA(opt.dataroot, shuffle_point = True)
elif opt.dataset_mode == 'animal':
    dataset = SMAL_DATA(opt.dataroot, shuffle_point = True)
else:
    raise ValueError("|dataset_mode| is invalid")
dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize, shuffle=True, num_workers=int(opt.nThreads), drop_last=opt.isTrain)

  
# create tool for counting iterations
iter_counter = IterationCounter(opt, len(dataloader))

# create trainer for our model
trainer = Ver2VerTrainer(opt)

# save root of the optputs
save_root = os.path.join(os.path.dirname(opt.checkpoints_dir), 'output', opt.dataset_mode)
if not os.path.exists(save_root):
    os.makedirs(save_root)


for epoch in iter_counter.training_epochs(): 
    iter_counter.record_epoch_start(epoch)
    for i, data_i in enumerate(dataloader, start=iter_counter.epoch_iter):
        iter_counter.record_one_iteration()

        # get data
        identity_points, pose_points, gt_points, id_face, pose_face = data_i

        # training
        trainer.train_model(identity_points, pose_points, gt_points, id_face)

        # print loss
        if iter_counter.needs_printing():  
            losses = trainer.get_latest_losses()
            try:
                print_current_errors(opt, epoch, iter_counter.epoch_iter,
                                                losses, iter_counter.time_per_iter)
            except OSError as err:
                print(err)

        # save mesh
        if iter_counter.needs_displaying():  
            try:
                pymesh.save_mesh_raw(save_root + '/' + str(epoch) + '_' + str(iter_counter.total_steps_so_far) + '_id.obj', 
                                                identity_points[0,:,:].cpu().numpy(),id_face[0,:,:].cpu().numpy())
                pymesh.save_mesh_raw(save_root + '/' + str(epoch) + '_' + str(iter_counter.total_steps_so_far) + '_pose.obj', 
                                                pose_points[0,:,:].cpu().numpy(),pose_face[0,:,:].cpu().numpy()) 
                pymesh.save_mesh_raw(save_root + '/' + str(epoch) + '_' + str(iter_counter.total_steps_so_far) + '_warp.obj', 
                                                trainer.out['warp_out'][0,:,:].cpu().detach().numpy(),id_face[0,:,:].cpu().numpy())      
                pymesh.save_mesh_raw(save_root + '/' + str(epoch) + '_' + str(iter_counter.total_steps_so_far) + '_out.obj', 
                                                trainer.get_latest_generated().data[0,:,:].cpu().detach().numpy().transpose(1,0),id_face[0,:,:].cpu().numpy())                            
            except OSError as err:
                print(err)

        if iter_counter.needs_saving():
            print('saving the latest model (epoch %d, total_steps %d)' %
                (epoch, iter_counter.total_steps_so_far))
            try:
                trainer.save('latest')
                iter_counter.record_current_iter()
            except OSError as err:
                print(err)

    trainer.update_learning_rate(epoch)
    iter_counter.record_epoch_end()

    if epoch % opt.save_epoch_freq == 0 or \
       epoch == iter_counter.total_epochs:
        print('saving the model at the end of epoch %d, iters %d' %
              (epoch, iter_counter.total_steps_so_far))
        try:
            trainer.save('latest')
            trainer.save(epoch)
        except OSError as err:
            print(err)

print('Training was successfully finished.')
