import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   

import sys
import argparse
import time

import cProfile
import re


parser = argparse.ArgumentParser()
parser.add_argument('-mode', type=str, help='rgb or flow', default='rgb')
parser.add_argument('-dataset', help='multithumos or charades', type=str, default='multithumos')
parser.add_argument('-save_model', type=str)
parser.add_argument('-root_train', type=str,
                    default='/mnt/data_a/alex/PyTorch_I3D/thumos/validation/')
parser.add_argument('-root_eval', type=str,
                    default='/mnt/data_a/alex/PyTorch_I3D/thumos/test/subset_test/')
parser.add_argument('-train_split', type=str,
                    default='/mnt/data_a/alex/PyTorch_I3D/thumos/validation/validation_thumos.json')
parser.add_argument('-eval_split', type=str,
                    default='/mnt/data_a/alex/PyTorch_I3D/thumos/test/test_thumos.json')
parser.add_argument('-snippets', type=int, default=64)
parser.add_argument('-batch_size', type=int, default=4)
parser.add_argument('-batch_size_eval', type=int, default=4)
parser.add_argument('-saving_steps', type=int, default=5000)
parser.add_argument('-num_steps_per_update', type=int, default=1)
parser.add_argument('-num_classes', type=int, default=65)
parser.add_argument('-max_steps', type=int, default=1e8)
parser.add_argument('-init_lr', type=float, default=0.1)
parser.add_argument('-use_cls', type=bool, default=False)
parser.add_argument('-crf', type=bool, default=False)
parser.add_argument('-num_updates_crf', type=int, default=1)
parser.add_argument('-pairwise_cond_crf', type=bool, default=False)
parser.add_argument('-reg_crf', type=float, default=-1)
parser.add_argument('-reg_type', type=str, default='l2')




args = parser.parse_args()


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable

import torchvision
from torchvision import datasets, transforms
import videotransforms


import numpy as np

from pytorch_i3d import InceptionI3d

dataset = args.dataset
if dataset=='multithumos':
    from multithumos_dataset import Multithumos as Dataset
elif dataset=='charades':
    from charades_dataset import Charades as Dataset

from metrics import ap_calculator, map_calculator
from utils import pt_var_to_numpy, last_checkpoint, Cumulator, get_reg_loss, get_param_crf

from tensorboard_logger import configure, log_value
import subprocess


def run(init_lr=0.1,
        max_steps=1e8,
        mode='rgb',
        dataset='thumos',
        root_train='/mnt/data_a/alex/PyTorch_I3D/thumos/validation/',
        root_eval='/mnt/data_a/alex/PyTorch_I3D/thumos/test/',
        train_split='/mnt/data_a/alex/PyTorch_I3D/thumos/validation/validation_thumos.json',
        eval_split='/mnt/data_a/alex/PyTorch_I3D/thumos/test/test_thumos.json',
        batch_size=4,
        batch_size_eval=4,
        save_model='',
        snippets=64,
        saving_steps=5000,
        num_steps_per_update=1,
        num_classes=65,
        crf=False,
        num_updates_crf=1,
        reg_crf=-1,
        use_cls=False,
        pairwise_cond_crf=False,
        reg_type='l2'):

    # setup dataset
    train_transforms = transforms.Compose([videotransforms.RandomCrop(224),
                                           videotransforms.RandomHorizontalFlip(),
    ])
    test_transforms = transforms.Compose([videotransforms.CenterCrop(224)])

    dataset = Dataset(train_split, 'training', root_train, mode, snippets, train_transforms)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True, drop_last=True)

    val_dataset = Dataset(eval_split, 'testing', root_eval, mode, snippets, test_transforms)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size_eval, shuffle=True, num_workers=8, pin_memory=True, drop_last=True)    

    dataloaders = {'train': dataloader, 'val': val_dataloader}
    datasets = {'train': dataset, 'val': val_dataset}

    
    # setup model
    steps = 0
    epoch = 0
    if not os.path.exists(args.save_model):
        subprocess.call('mkdir ' + args.save_model, shell=True)
    configure(args.save_model + "tensorboard_logger", flush_secs=5)

    # resume the training or load the pre-trained I3D
    checkpoint=-1
    try:
        checkpoint = last_checkpoint(args.save_model)
    except:
        print("Loading the pre-trained I3D")
        subprocess.call('mkdir ' + args.save_model + "/tensorboard_logger", shell=True)
        if mode == 'flow':
            i3d = InceptionI3d(400, in_channels=2, use_crf=crf, num_updates_crf=num_updates_crf, pairwise_cond_crf=pairwise_cond_crf)
            total_dict = i3d.state_dict()
            partial_dict = torch.load('models/flow_imagenet.pt')
            total_dict.update(partial_dict)
            i3d.load_state_dict(total_dict)

        else:
            i3d = InceptionI3d(400, in_channels=3, use_crf=crf, num_updates_crf=num_updates_crf, pairwise_cond_crf=pairwise_cond_crf)
            total_dict = i3d.state_dict()
            partial_dict = torch.load('models/rgb_imagenet.pt')
            total_dict.update(partial_dict)
            i3d.load_state_dict(total_dict)

        i3d.replace_logits(num_classes)

    if (checkpoint!=-1):
        if mode == 'flow':
            i3d = InceptionI3d(num_classes, in_channels=2, use_crf=crf, num_updates_crf=num_updates_crf, pairwise_cond_crf=pairwise_cond_crf)

        else:
            i3d = InceptionI3d(num_classes, in_channels=3, use_crf=crf, num_updates_crf=num_updates_crf, pairwise_cond_crf=pairwise_cond_crf)

        
        i3d.load_state_dict(torch.load(args.save_model + checkpoint))
        steps = int(checkpoint[:-3])
        if dataset=='thumos':
            epoch = int(steps*snippets*batch_size*num_steps_per_update / 1214016)
        else:
            epoch = int(steps*snippets*batch_size*num_steps_per_update / 5482688)

        
    # push the pipeline on multiple gpus if possible
    i3d.cuda()
    i3d = nn.DataParallel(i3d)
    
    # setup optimizer
    lr = init_lr
    optimizer = optim.SGD(i3d.parameters(), lr=lr, momentum=0.9, weight_decay=0.0000001)
    lr_sched = optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=[1000], gamma=0.1)
    if steps>0:
        for i in range(steps):
            lr_sched.step()

    # train the model
    while steps < max_steps:
        epoch += 1
        print('-' * 10)
        print('Epoch {}'.format(epoch))
        print('Step {}/{}'.format(steps, max_steps))
        print('-' * 10)

        # each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                print('Entering training loop...')
                i3d.train()
            else:
                print('Entering validation loop...')
                i3d.eval()
                time_init_eval = time.time()

            cumul_pred = Cumulator(num_classes)
            cumul_labels = Cumulator(num_classes)
                
            tot_loss = 0.0
            tot_loc_loss = 0.0
            tot_cls_loss = 0.0
            tot_loss_updt = 0.0
            tot_loc_loss_updt = 0.0
            tot_cls_loss_updt = 0.0
            tot_reg_loss_updt = 0.0
            num_iter = 0
            optimizer.zero_grad()
            count_batch = 0
            gap_train = 0

            print("Losses initialized to 0")
            
            # Iterate over data.
            for data in dataloaders[phase]:
                time_init_batch = time.time()
                count_batch += 1
                num_iter += 1
                # get the inputs
                inputs, labels = data

                # wrap them in Variable
                inputs = Variable(inputs.cuda())
                t = inputs.size(2)
                labels = Variable(labels.cuda())

                # forward
                if crf:
                    per_frame_logits_ante_crf, per_frame_logits = i3d(inputs)
                else:    
                    per_frame_logits = i3d(inputs)

                # upsample to input size
                per_frame_logits = F.upsample(per_frame_logits, t, mode='linear')
                if crf:
                    per_frame_logits_ante_crf = F.upsample(per_frame_logits_ante_crf, t, mode='linear')
                
                # accumulate predictions and ground truths
                pred_np = pt_var_to_numpy(nn.Sigmoid()(per_frame_logits))
                cumul_pred.append(pred_np)
                labels_np = pt_var_to_numpy(labels)
                cumul_labels.append(labels_np)

                # compute localization loss
                if crf:
                    loc_loss = F.binary_cross_entropy_with_logits(per_frame_logits, labels) + F.binary_cross_entropy_with_logits(per_frame_logits_ante_crf, labels)
                else:
                    loc_loss = F.binary_cross_entropy_with_logits(per_frame_logits, labels)
                tot_loc_loss += loc_loss.data[0]
                tot_loc_loss_updt += loc_loss.data[0]

                # compute classification loss (with max-pooling along time B x C x T)
                if crf:
                    cls_loss = F.binary_cross_entropy_with_logits(torch.max(per_frame_logits, dim=2)[0], torch.max(labels, dim=2)[0]) + F.binary_cross_entropy_with_logits(torch.max(per_frame_logits_ante_crf, dim=2)[0], torch.max(labels, dim=2)[0])
                else:
                    cls_loss = F.binary_cross_entropy_with_logits(torch.max(per_frame_logits, dim=2)[0], torch.max(labels, dim=2)[0])
                tot_cls_loss += cls_loss.data[0]
                tot_cls_loss_updt += cls_loss.data[0]

                # compute regularization loss for the crf module
                if crf and (reg_crf>0 and not pairwise_cond_crf) :
                    reg_loss = get_reg_loss(i3d, 'crf', reg_type)
                    tot_reg_loss_updt += reg_loss.data[0]
                elif crf and (reg_crf>0 and pairwise_cond_crf) :
                    reg_loss = get_reg_loss(i3d, 'psi_0', reg_type) + get_reg_loss(i3d, 'psi_1', reg_type)
                    tot_reg_loss_updt += reg_crf*reg_loss.data[0]
                else:
                    reg_loss=0

                # put all the losses together
                if use_cls:
                    loss = (0.5*loc_loss + 0.5*cls_loss + reg_crf*reg_loss)/num_steps_per_update
                else:
                    loss = (loc_loss + reg_crf*reg_loss)/num_steps_per_update

                tot_loss += loss.data[0]
                tot_loss_updt += loss.data[0]
                loss.backward()

                if num_iter == num_steps_per_update and phase == 'train':
                    steps += 1
                    num_iter = 0
                    optimizer.step()
                    optimizer.zero_grad()
                    lr_sched.step()
                    examples_processed_updt = num_steps_per_update*batch_size*snippets
                    examples_processed_tot = count_batch*batch_size*snippets
                    map_train = map_calculator(cumul_pred.accumuled[1:], cumul_labels.accumuled[1:])
                    gap_train = ap_calculator(cumul_pred.accumuled[1:].flatten(), cumul_labels.accumuled[1:].flatten())
                    print('TRAINING - Epoch: {} Step: {} Examples processed {} Loc Loss: {:.6f} Cls Loss: {:.6f} Tot Loss: {:.6f} Reg Loss: {:.6f} mAP: {:.6f} GAP: {:.6f}'.format(epoch, steps, examples_processed_tot, tot_loc_loss_updt/examples_processed_updt, tot_cls_loss_updt/examples_processed_updt, tot_loss_updt/(batch_size*snippets), reg_crf*tot_reg_loss_updt/examples_processed_updt, map_train, gap_train))
                    log_value('Training_loc_loss', tot_loc_loss_updt/examples_processed_updt, steps)
                    log_value('Training_cls_loss', tot_cls_loss_updt/examples_processed_updt, steps)
                    log_value('Training_reg_loss', tot_reg_loss_updt/examples_processed_updt, steps)
                    log_value('Training_tot_loss', tot_loss_updt/(batch_size*snippets), steps)
                    log_value('Training_mAP', map_train, steps)
                    log_value('Training_GAP', gap_train, steps)
                    tot_loss_updt, tot_loc_loss_updt, tot_cls_loss_updt, tot_reg_loss_updt = 0.0, 0.0, 0.0, 0.0
                    cumul_pred.clear()
                    cumul_labels.clear()
                    cumul_pred = Cumulator(num_classes)
                    cumul_labels = Cumulator(num_classes)

                if ((steps % saving_steps)==0) & (phase=='train') & (num_iter==0):
                    # save model
                    print("EPOCH: {} Step: {} - Saving model...".format(epoch, steps))
                    torch.save(i3d.module.state_dict(), save_model+str(steps).zfill(6)+'.pt')
                    tot_loss = tot_loc_loss = tot_cls_loss = 0.
                    if crf:
                        psi_0, psi_1 = get_param_crf(i3d)
                        np.save(save_model + 'psi_0_' + str(steps), pt_var_to_numpy(psi_0))
                        np.save(save_model + 'psi_1_' + str(steps), pt_var_to_numpy(psi_1))
                
                if phase == 'val':
                    time_end_batch = time.time()
                    examples_processed_tot = count_batch*batch_size_eval*snippets
                    print('EVAL - Epoch: {} Step: {} Examples processed {} - Time for batch: {}'.format(epoch, steps, examples_processed_tot, time_end_batch - time_init_batch))
                    log_value('Evaluation time', time_end_batch - time_init_batch, examples_processed_tot)

            

            if phase=='val':
                examples_processed_tot = count_batch*batch_size_eval*snippets
                map_val = map_calculator(cumul_pred.accumuled[1:], cumul_labels.accumuled[1:])
                gap_val = ap_calculator(cumul_pred.accumuled[1:].flatten(), cumul_labels.accumuled[1:].flatten())
                time_end_eval = time.time()
                print('EVAL - Epoch: {} Step: {} Loc Loss: {:.6f} Cls Loss: {:.6f} Tot Loss: {:.6f} mAP: {:.4f} GAP: {:.4f} Total time: {}'.format(epoch, steps, tot_loc_loss/examples_processed_tot, tot_cls_loss/examples_processed_tot, tot_loss_updt*num_steps_per_update/examples_processed_tot, map_val, gap_val, time_end_eval-time_init_eval))
                log_value('Validation_subset_loc_loss', tot_loc_loss/examples_processed_tot, steps)
                log_value('Validation_subset_cls_loss', tot_cls_loss/examples_processed_tot, steps)
                log_value('Validation_subset_tot_loss', tot_loss_updt*num_steps_per_update/examples_processed_tot)
                log_value('Validation_subset_mAP', map_val, steps)
                log_value('Validation_subset_GAP', gap_val, steps)
                cumul_pred.clear()
                cumul_labels.clear()

        
    


if __name__ == '__main__':
    # need to add argparse
    run(init_lr=args.init_lr,
        max_steps=args.max_steps,
        mode=args.mode,
        dataset=args.dataset,
        root_train=args.root_train,
        root_eval=args.root_eval,
        train_split=args.train_split,
        eval_split=args.eval_split,
        save_model=args.save_model,
        snippets=args.snippets,
        batch_size=args.batch_size,
        batch_size_eval=args.batch_size_eval,
        saving_steps=args.saving_steps,
        num_steps_per_update=args.num_steps_per_update,
        num_classes=args.num_classes,
        crf=args.crf,
        num_updates_crf=args.num_updates_crf,
        reg_crf=args.reg_crf,
        use_cls=args.use_cls,
        pairwise_cond_crf=args.pairwise_cond_crf,
        reg_type=args.reg_type)

