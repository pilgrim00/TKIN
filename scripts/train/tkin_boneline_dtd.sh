#cd /apdcephfs/private_jiaxianchen/PISE_out
#cp  ./tkin_k1_dtd/fashion/latest_net_{D,G,D_dtd}.pth  ./tkin_bonelinedtd_k1/fashion/ 
#cp  ./12_10_puredtd/fashion/latest_net_{D,G,D_dtd}.pth  ./tkin_bonelinedtd_k3/fashion/
#cp  ./tkin_k5_dtd/fashion/20000_net_{D,G,D_dtd}.pth  ./tkin_bonelinedtd_k5/fashion/
#cp  ./tkin_k7_dtd/fashion/20000_net_{D,G,D_dtd}.pth  ./tkin_bonelinedtd_k7/fashion/
#exit
#本地测试行得通
python3 -m torch.distributed.launch --nproc_per_node=1  train.py --name=fashion --model=bonelinedtd --dist=1  \
--dataset_mode dtdfashion  --batchSize=1 --nThreads=0  --lr 1e-4  --print_freq 100  --epochs 200  \
--display_freq=100 --use_rec=0 --point_line=1 \
--continue_train --save_latest_freq  10000   --save_iters_freq 10000  --optim 'Adam'  --niter 200000  --ker_size 1 \
--checkpoints_dir '/mnt/private_jiaxianchen/PISE_out/tkin_bonelinedtd_k1' 
python3 -m torch.distributed.launch --nproc_per_node=4  train.py --name=fashion --model=bonelinedtd --dist=1  \
--dataset_mode dtdfashion  --batchSize=8 --nThreads=16  --lr 1e-4  --print_freq 100  --epochs 200  \
--display_freq=500 --use_rec=0 --point_line=1 \
--continue_train --save_latest_freq  10000   --save_iters_freq 10000  --optim 'Adam'  --niter 200000  --ker_size 5 \
--checkpoints_dir '/mnt/private_jiaxianchen/PISE_out/tkin_bonelinedtd_k5' >> /mnt/private_jiaxianchen/PISE_out/tkin_bonelinedtd_k5/out.txt 