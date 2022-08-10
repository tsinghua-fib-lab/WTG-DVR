# DVR: Micro-Video Recommendation Optimizing Watch-Time-Gain under Duration Bias

This is the official implementation of our MM'22 paper:  

Yu Zheng, Chen Gao, Jingtao Ding, Lingling Yi, Depeng Jin, Yong Li, Meng Wang, **DVR: Micro-Video Recommendation Optimizing Watch-Time-Gain under Duration Bias**, In Proceedings of the ACM Multimedia 2022.

The code is tested under a Linux desktop with TensorFlow 2.3.0 and Python 3.7.9.

## Datasets
Unzip the compressed data files in `examples/data/` with the following commands:
```
cd ./examples/data/
unzip kuaishou_video_debias.csv.zip
unzip wechat_video_debias.csv.zip
```

## Model Training
Use the following commands for model training.

To train a basic DeepFM model on `Kuaishou` dataset: 
```
cd ./examples
python run_video_debias.py --name kuaishou-deepfm --model DeepFM --dataset kuaishou 
```

To train a DVR- version of DeepFM model on `Kuaishou` dataset: 
```
cd ./examples
python run_video_debias.py --name kuaishou-deepfm-dvrminus --model DeepFM --dataset kuaishou --post_transform
```

To train a DVR version of DeepFM model on `Kuaishou` dataset: 
```
cd ./examples
python run_video_debias.py --name kuaishou-deepfm-dvr --model DeepFM --dataset kuaishou --train_target gain --remove_duration_feature --disentangle --disentangle_loss_weight 0.1
```

You can check the FLAGS in `examples/run_video_debias.py` to explore other experimental settings.

## Note

The implemention is based on *[DeepCTR](https://github.com/shenweichen/DeepCTR)*.