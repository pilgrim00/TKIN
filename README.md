# TKIN
This is for our paper "[Exploring Kernel-based Texture Transfer for Pose-guided Person Image Generation](10.1109/TMM.2022.3221351)" which is accepted by the IEEE Transactions on Multimedia(TMM).
# Requirement
```
conda create -n tkin python=3.6
conda install pytorch=1.2 cudatoolkit=10.0 torchvision
pip install scikit-image pillow pandas tqdm dominate natsort 
```
# Data
Data preparation for images , keypoints and parsing maps can follow [PISE](https://github.com/Zhangjinso/PISE),which are all based on the DeepFashion-inshop dataset.

Additionally, [DTD dataset](https://paperswithcode.com/dataset/dtd) if needed can be downloaded from [baidu](https://pan.baidu.com/s/11HTqi2esY9nMorzcSi1qkg)(fectch code: 4z7r)

# Citation

If you use this code, please cite our paper.

```
@ARTICLE{9944889,  author={Chen, Jiaxiang and Fan, Jiayuan and Ye, Hancheng and Li, Jie and Liao, Yongbing and Chen, Tao},  
journal={IEEE Transactions on Multimedia},   
title={Exploring Kernel-based Texture Transfer for Pose-guided Person Image Generation},   
year={2022},  
pages={1-14},  
doi={10.1109/TMM.2022.3221351}}
```
# Acknowledgments
Our code is based on the [GFLA](https://github.com/RenYurui/Global-Flow-Local-Attention) and [PISE](https://github.com/Zhangjinso/PISE).
