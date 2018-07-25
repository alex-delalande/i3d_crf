# CRF for multi-label video classification

## Overview

This repository contains the PyTorch implementation of the CRF structure for multi-label video classification. It uses I3D pre-trained models as base classifiers (I3D is reported in the paper "[Quo Vadis,
Action Recognition? A New Model and the Kinetics
Dataset](https://arxiv.org/abs/1705.07750)" by Joao Carreira and Andrew
Zisserman).

This code is based on Deepmind's [Kinetics-I3D](https://github.com/deepmind/kinetics-i3d) and on AJ Piergiovanni's [PyTorch implementation](https://github.com/piergiaj/pytorch-i3d) of the I3D pipeline.



## Requirement

This code requieres [PyTorch](https://pytorch.org/) and [tensorboard_logger](https://github.com/TeamHG-Memex/tensorboard_logger).

## Training I3D + semi/fully-CRF end-to-end

This pipeline uses Deepmind's pretrained I3D models (pretrained on ImageNet and Kinetics, see [Kinetics-I3D](https://github.com/deepmind/kinetics-i3d) for details). These are the models denoted as rgb_imagenet.pt and flow_imagenet.pt found in the directory **models/**.


### Base model (I3D)

The base model can be trained using the following command:

```
python train_i3d.py -dataset 'charades' -mode 'flow' -save_model 'path_to_saving_directory' -root_train 'path_to_flow_training_data' -train_split 'path_to_train_charades.json' -root_eval 'path_to_flow_evaluation_data' -eval_split 'path_to_test_charades.json' -snippets 64 -batch_size 4 -batch_size_eval 4 -saving_steps 5000 -num_steps_per_update 1 -num_classes 157 -init_lr 0.1 -use_cls True
```

Dataset is either 'charades' or 'multithumos', mode is either 'flow' or 'rgb'.


### Semi-CRF model

To add the semi-CRF structure, add '-crf True' and the regularization value wanted as follows:
```
python train_i3d.py -dataset 'charades' -mode 'rgb' -save_model 'path_to_saving_directory' -root_train 'path_to_rgb_training_data' -train_split 'path_to_train_charades.json' -root_eval 'path_to_rgb_evaluation_data' -eval_split 'path_to_test_charades.json' -snippets 64 -batch_size 4 -batch_size_eval 4 -saving_steps 5000 -num_steps_per_update 1 -num_classes 157 -init_lr 0.1 -use_cls True -crf True -reg_crf 1e-4
```

### Fully-CRF model

To add the fully-CRF structure, add '-conditional_crf True' and the regularization value wanted as follows:
```
python train_i3d.py -mode 'rgb' -save_model 'path_to_saving_directory' -root_train 'path_to_rgb_training_data' -train_split 'path_to_train_thumos.json' -root_eval 'path_to_rgb_evaluation_data' -eval_split 'path_to_test_thumos.json' -snippets 64 -batch_size 4 -batch_size_eval 4 -saving_steps 5000 -num_steps_per_update 1 -num_classes 65 -init_lr 0.1 -crf True -use_cls True -conditional_crf True -reg_crf 1e-3
```


## Testing

### One-stream (RGB of Optical-flow)

```
python eval_i3d.py -dataset 'charades' -mode 'rgb' -save_model 'path_to_saving_directory' -root_eval 'path_to_rgb_evaluation_data' -eval_split  'path_to_test_charades.json' -snippets 64 -batch_size_eval 1 -num_classes 157 -crf True -eval_checkpoint 750000
```


### Two-stream

```
python eval_i3d_2_streams.py -dataset 'charades' -save_model_rgb 'path_to_rgb_saving_directory' -save_model_flow 'path_to_flow_saving_directory' -root_eval_rgb 'path_to_rgb_test_data' -root_eval_flow 'path_to_flow_test_data' -eval_split 'path_to_test_charades.json' -snippets 64 -batch_size_eval 1 -crf True -num_classes 157 -eval_checkpoint_rgb 500000 -eval_checkpoint_flow 500000
```


## Visualization of logged events
To visualize logged event through TensorBoard, use:
```
tensorboard --logdir=path_to_saving_directory/tensorboard_logger
```

For 2-Streams events (after use of eval_i3d_2_streams.py):
```
tensorboard --logdir=path_to_rgb_saving_directory/tensorboard_logger_2_streams
```

