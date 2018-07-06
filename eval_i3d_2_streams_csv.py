import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
#os.environ["CUDA_VISIBLE_DEVICES"]='0,1,2,3'
import sys
import argparse
import time

parser = argparse.ArgumentParser()
parser.add_argument('-save_model_rgb', type=str)
parser.add_argument('-save_model_flow', type=str)
parser.add_argument('-root_eval_rgb', type=str)
parser.add_argument('-root_eval_flow', type=str)
parser.add_argument('-eval_split', type=str,
                    default='/mnt/data_a/alex/PyTorch_I3D/thumos/test/test_thumos.json')
parser.add_argument('-snippets', type=int, default=64)
parser.add_argument('-batch_size_eval', type=int, default=4)
parser.add_argument('-num_classes', type=int, default=65)
parser.add_argument('-crf', type=bool, default=False)
parser.add_argument('-num_updates_crf', type=int, default=1)
parser.add_argument('-eval_checkpoint_rgb', type=int, default=-1)
parser.add_argument('-eval_checkpoint_flow', type=int, default=-1)
parser.add_argument('-pairwise_cond_crf', type=bool, default=False)
parser.add_argument('-dataset', help='multithumos or charades', type=str, default='multithumos')


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


def run(root_eval_rgb,
        root_eval_flow,
        eval_split='/mnt/data_a/alex/PyTorch_I3D/thumos/test/test_thumos.json',
        batch_size_eval=2,
        save_model_rgb='',
        save_model_flow='',
        snippets=64,
        num_classes=65,
        crf=False,
        pairwise_cond_crf=False,
        num_updates_crf=1,
        eval_checkpoint_rgb=-1,
        eval_checkpoint_flow=-1):
    # setup dataset
    test_transforms = transforms.Compose([videotransforms.CenterCrop(224)])

    val_dataset_rgb = Dataset(eval_split, 'training', root_eval_flow, 'flow', snippets, test_transforms) # 'testing' !!!!!!!
    val_dataset_rgb.mode = 'rgb'
    val_dataset_rgb.root = root_eval_rgb
    val_dataloader_rgb = torch.utils.data.DataLoader(val_dataset_rgb, batch_size=batch_size_eval, shuffle=False, num_workers=4, pin_memory=True) 

    val_dataset_flow = Dataset(eval_split, 'training', root_eval_flow, 'flow', snippets, test_transforms)
    val_dataloader_flow = torch.utils.data.DataLoader(val_dataset_flow, batch_size=batch_size_eval, shuffle=False, num_workers=4, pin_memory=True)        

    dataloaders = {'val_rgb': val_dataloader_rgb, 'val_flow': val_dataloader_flow}
    datasets = {'val_rgb': val_dataset_rgb, 'val_flow': val_dataset_flow}

    # load the targeted models
    i3d_rgb = InceptionI3d(num_classes, in_channels=3, use_crf=crf, num_updates_crf=num_updates_crf, pairwise_cond_crf=pairwise_cond_crf)
    i3d_flow = InceptionI3d(num_classes, in_channels=2, use_crf=crf, num_updates_crf=num_updates_crf, pairwise_cond_crf=pairwise_cond_crf)

    if eval_checkpoint_rgb<0:
        checkpoint_rgb = last_checkpoint(args.save_model_rgb)
    else:
        checkpoint_rgb = str(eval_checkpoint_rgb).zfill(6)+'.pt'
    
    if eval_checkpoint_flow<0:
        checkpoint_flow = last_checkpoint(args.save_model_flow)
    else:
        checkpoint_flow = str(eval_checkpoint_flow).zfill(6)+'.pt'

    i3d_rgb.load_state_dict(torch.load(args.save_model_rgb + checkpoint_rgb))
    steps_rgb = int(checkpoint_rgb[:-3])

    i3d_flow.load_state_dict(torch.load(args.save_model_flow + checkpoint_flow))
    steps_flow = int(checkpoint_flow[:-3])
    
    i3d_rgb.cuda()
    i3d_rgb = nn.DataParallel(i3d_rgb)

    i3d_flow.cuda()
    i3d_flow = nn.DataParallel(i3d_flow)
    subprocess.call('mkdir ' + args.save_model_rgb + "/tensorboard_logger_2_streams", shell=True)
    configure(args.save_model_rgb + "tensorboard_logger_2_streams", flush_secs=5)

    # Preparing csv files for containing the predictions and ground truths
    path_csv_pred = args.save_model_rgb+'/pred_eval_2_streams_step_rgb_' + str(steps_rgb) + '_step_flow_' + str(steps_flow) + '.csv'
    if os.path.exists(path_csv_pred):
        subprocess.call('rm -r ' + path_csv_pred, shell=True)

    path_csv_labels = args.save_model_rgb+'/labels_eval_2_streams_step_rgb_' + str(steps_rgb) + '_step_flow_' + str(steps_flow) + '.csv'
    if os.path.exists(path_csv_labels):
        subprocess.call('rm -r ' + path_csv_labels, shell=True)

    csvfile_pred = open(path_csv_pred, 'a')
    cumul_pred = csv.writer(csvfile_pred, delimiter=',')
    csvfile_label = open(path_csv_labels, 'a')
    cumul_label = csv.writer(csvfile_label, delimiter=',')


    print('Entering validation loop...')
    i3d_rgb.train(False)
    i3d_flow.train(False)

    time_init_eval = time.time()

    num_iter = 0
    count_batch = 0

    len_data = len(dataloaders['val_rgb'])
            
    # Iterate over data.
    for (data_rgb, data_flow) in zip(dataloaders['val_rgb'], dataloaders['val_flow']):
        time_init_batch = time.time()
        count_batch += 1
        num_iter += 1
        # get the inputs
        inputs_rgb, labels_rgb = data_rgb
        inputs_flow, labels_flow = data_flow


        # wrap them in Variable
        inputs_rgb = Variable(inputs_rgb.cuda())
        t = inputs_rgb.size(2)
        labels_rgb = Variable(labels_rgb.cuda())
        inputs_flow = Variable(inputs_flow.cuda())
        t = inputs_flow.size(2)
        labels_flow = Variable(labels_flow.cuda())

        if crf:
            _, per_frame_logits_rgb = i3d_rgb(inputs_rgb)
            _, per_frame_logits_flow = i3d_flow(inputs_flow)
        else:    
            per_frame_logits_rgb = i3d_rgb(inputs_rgb)
            per_frame_logits_flow = i3d_flow(inputs_flow)
        # upsample to input size
        per_frame_logits_rgb = F.upsample(per_frame_logits_rgb, t, mode='linear')
        per_frame_logits_flow = F.upsample(per_frame_logits_flow, t, mode='linear')

        pred_np = pt_var_to_numpy(nn.Sigmoid()((per_frame_logits_rgb+per_frame_logits_flow)/2))
        pred_np = np.transpose(pred_np, [0,2,1]).reshape([-1, num_classes])
        for l in pred_np:
            cumul_pred.writerow(l)
        pred_np = []

        labels_np = pt_var_to_numpy(labels_rgb)
        labels_np = np.transpose(labels_np, [0,2,1]).reshape([-1, num_classes])
        for l in labels_np:
            cumul_label.writerow(l)
        labels_np = []

        time_end_batch = time.time()
        examples_processed_tot = count_batch*batch_size_eval*snippets
        print('EVAL - Examples processed {} - Time for batch: {}'.format(examples_processed_tot, time_end_batch - time_init_batch))
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
    print("EVAL - Step RGB: {} Step FLOW: {} mAP = {} - Computed in {}".format(steps_rgb, steps_flow, map_eval, time.time()-t0))
    print("Computing GAP...")
    t0 = time.time()
    gap_eval = average_precision_score(labels, pred, average="micro")
    print("EVAL - Step RGB: {} Step FLOW: {} GAP = {} - Computed in {}".format(steps_rgb, steps_flow, gap_eval, time.time()-t0))

    log_value('Validation_mAP_RGB', map_eval, steps_rgb)
    log_value('Validation_mAP_FLOW', map_eval, steps_flow)
    log_value('Validation_GAP_RGB', gap_eval, steps_rgb)
    log_value('Validation_GAP_FLOW', gap_eval, steps_flow)

    pred, labels = [], []


    


if __name__ == '__main__':
    # need to add argparse
    run(root_eval_rgb=args.root_eval_rgb,
        root_eval_flow=args.root_eval_flow,
        eval_split=args.eval_split,
        save_model_rgb=args.save_model_rgb,
        save_model_flow=args.save_model_flow,
        snippets=args.snippets,
        batch_size_eval=args.batch_size_eval,
        num_classes=args.num_classes,
        crf=args.crf,
        pairwise_cond_crf=args.pairwise_cond_crf,
        num_updates_crf=args.num_updates_crf,
        eval_checkpoint_rgb=args.eval_checkpoint_rgb,
        eval_checkpoint_flow=args.eval_checkpoint_flow)
