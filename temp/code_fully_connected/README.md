# Fully-connected experiments


Here is a way to launch this code:

* Go on DGX and select PyTorch env
```
ssh alex@10.217.128.158
```
password: alex

```
export PATH=/home/alex/anaconda3/bin:$PATH
export PATH=/usr/local/cuda-9.0/bin${PATH:+:${PATH}}$
export LD_LIBRARY_PATH=/usr/local/cuda-9.0/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
source activate pytorch
```

```
cd /raid/alex/PyTorch_I3D/code_fully_connected/
```


Then launch  train_i3d.py. **Be sure to be in th “code_fully_connected/” folder**.

For instance on Charades RGB the code is:
```
python train_i3d.py -dataset 'charades' -mode 'rgb' -save_model '/raid/alex/charades/rgb/models/fully_connected_model_reg_1e-4/' -root_train '/raid/alex/charades/rgb/train/' -train_split '/raid/alex/charades/train_charades.json' -root_eval '/raid/alex/charades/rgb/subset_test/' -eval_split '/raid/alex/charades/test_charades.json' -snippets 64 -batch_size 4 -batch_size_eval 4 -saving_steps 2000 -num_steps_per_update 1 -num_classes 157 -init_lr 0.1 -use_cls True -crf True -reg_crf 1e-4
```

