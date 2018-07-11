import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
#os.environ["CUDA_VISIBLE_DEVICES"]='0,1,2,3'
import sys
import argparse
import time

parser = argparse.ArgumentParser()
parser.add_argument('-mode', type=str, help='rgb or flow', default='rgb')
parser.add_argument('-dataset', help='multithumos or charades', type=str, default='multithumos')
parser.add_argument('-save_model', type=str)
parser.add_argument('-root_eval', type=str,
                    default='/mnt/data_a/alex/PyTorch_I3D/thumos/test/')
parser.add_argument('-eval_split', type=str,
                    default='/mnt/data_a/alex/PyTorch_I3D/thumos/test/test_thumos.json')
parser.add_argument('-snippets', type=int, default=64)
parser.add_argument('-batch_size_eval', type=int, default=4)
parser.add_argument('-eval_checkpoint', type=int, default=-1)
parser.add_argument('-num_classes', type=int, default=65)
parser.add_argument('-crf', type=bool, default=False)
parser.add_argument('-num_updates_crf', type=int, default=1)
parser.add_argument('-pairwise_cond_crf', type=bool, default=False)



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
from utils import pt_var_to_numpy, last_checkpoint, get_reg_loss

from tensorboard_logger import configure, log_value
import subprocess
import csv
from sklearn.metrics import average_precision_score


def run(mode='rgb',
        root_eval='/mnt/data_a/alex/PyTorch_I3D/thumos/test/',
        eval_split='/mnt/data_a/alex/PyTorch_I3D/thumos/test/test_thumos.json',
        batch_size_eval=10,
        save_model='',
        snippets=64,
        num_classes=65,
        crf=False,
        pairwise_cond_crf=False,
        num_updates_crf=1,
        eval_checkpoint=-1):
    # setup dataset
    test_transforms = transforms.Compose([videotransforms.CenterCrop(224)])

    val_dataset = Dataset(eval_split, 'testing', root_eval, mode, snippets, test_transforms)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size_eval, shuffle=False, num_workers=16, pin_memory=True)    

    dataloaders = {'val': val_dataloader}
    datasets = {'val': val_dataset}
    

    # load the targeted model
    if mode == 'flow':
        i3d = InceptionI3d(num_classes, in_channels=2, use_crf=crf, num_updates_crf=num_updates_crf, pairwise_cond_crf=pairwise_cond_crf)

    else:
        i3d = InceptionI3d(num_classes, in_channels=3, use_crf=crf, num_updates_crf=num_updates_crf, pairwise_cond_crf=pairwise_cond_crf)

    
    eval_checkpoint = args.eval_checkpoint
    if eval_checkpoint<0:
        checkpoint = last_checkpoint(args.save_model)
    else:
        checkpoint = str(eval_checkpoint).zfill(6)+'.pt'

    i3d.load_state_dict(torch.load(args.save_model + checkpoint))
    steps = int(checkpoint[:-3])
    
    i3d.cuda()
    i3d = nn.DataParallel(i3d)
    configure(args.save_model + "tensorboard_logger", flush_secs=5)



    print('-' * 10)
    print('Step {}'.format(steps))
    print('-' * 10)

    # Preparing csv files for containing the predictions and ground truths
    path_csv_pred = args.save_model+'/pred_eval_step_' + str(steps) + '.csv'
    if os.path.exists(path_csv_pred):
        subprocess.call('rm -r ' + path_csv_pred, shell=True)

    path_csv_labels = args.save_model+'/labels_eval_step_' + str(steps) + '.csv'
    if os.path.exists(path_csv_labels):
        subprocess.call('rm -r ' + path_csv_labels, shell=True)

    csvfile_pred = open(path_csv_pred, 'a')
    cumul_pred = csv.writer(csvfile_pred, delimiter=',')
    csvfile_label = open(path_csv_labels, 'a')
    cumul_label = csv.writer(csvfile_label, delimiter=',')


    print('Entering validation loop...')
    i3d.train(False)
    #i3d.eval()
    time_init_eval = time.time()

    num_iter = 0
    count_batch = 0
            
    # Iterate over data.
    for data in dataloaders['val']:
        time_init_batch = time.time()
        count_batch += 1
        num_iter += 1
        # get the inputs
        inputs, labels = data

        # wrap them in Variable
        inputs = Variable(inputs.cuda())
        t = inputs.size(2)
        labels = Variable(labels.cuda())

        if crf:
            _, per_frame_logits = i3d(inputs)
        else:    
            per_frame_logits = i3d(inputs)
        # upsample to input size
        per_frame_logits = F.upsample(per_frame_logits, t, mode='linear')

        pred_np = pt_var_to_numpy(nn.Sigmoid()(per_frame_logits))
        pred_np = np.transpose(pred_np, [0,2,1]).reshape([-1, num_classes])
        for l in pred_np:
            cumul_pred.writerow(l)
        pred_np = []

        labels_np = pt_var_to_numpy(labels)
        labels_np = np.transpose(labels_np, [0,2,1]).reshape([-1, num_classes])
        for l in labels_np:
            cumul_label.writerow(l)
        labels_np = []

        time_end_batch = time.time()
        examples_processed_tot = count_batch*batch_size_eval*snippets
        print('EVAL - Step: {} Examples processed {} - Time for batch: {}'.format(steps, examples_processed_tot, time_end_batch - time_init_batch))
        log_value('Evaluation time', time_end_batch - time_init_batch, examples_processed_tot)


    examples_processed_tot = count_batch*batch_size_eval*snippets
    time_end_eval = time.time()
    print("EVAl DONE in {}".format(time_end_eval - time_init_eval))
    csvfile_pred.close()
    csvfile_label.close()

    # Compute metrics
    print("Computing metrics. Loading labels..")
    t0 = time.time()
    labels = np.loadtxt(path_csv_labels, delimiter=",")
    print("Labels loaded in {}".format(time.time()-t0))
    print("Loading predictions...")
    t0 = time.time()
    pred = np.loadtxt(path_csv_pred, delimiter=",")
    print("Predictions loaded in {}".format(time.time()-t0))
    print("Computing mAP...")
    t0 = time.time()
    map_eval = average_precision_score(labels, pred, average="macro")
    print("EVAL - Step: {} mAP = {} - Computed in {}".format(steps, map_eval, time.time()-t0))
    print("Computing GAP...")
    t0 = time.time()
    gap_eval = average_precision_score(labels, pred, average="micro")
    print("EVAL - Step: {} GAP = {} - Computed in {}".format(steps, gap_eval, time.time()-t0))

    log_value('Validation_mAP', map_eval, steps)
    log_value('Validation_GAP', gap_eval, steps)

    pred, labels = [], []


        
    


if __name__ == '__main__':
    # need to add argparse
    run(mode=args.mode,
        root_eval=args.root_eval,
        eval_split=args.eval_split,
        save_model=args.save_model,
        snippets=args.snippets,
        batch_size_eval=args.batch_size_eval,
        num_classes=args.num_classes,
        crf=args.crf,
        pairwise_cond_crf=args.pairwise_cond_crf,
        num_updates_crf=args.num_updates_crf,
        eval_checkpoint=args.eval_checkpoint)
