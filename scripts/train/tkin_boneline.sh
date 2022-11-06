#本地测试行得通
#python3 -m torch.distributed.launch --nproc_per_node=1  train.py --name=fashion --model=boneline --dist=1  \
#--batchSize=1 --nThreads=0  --lr 1e-4  --print_freq 100  --epochs 200  \
#--display_freq=100 --use_rec=0 --point_line=1 \
#--continue_train --save_latest_freq  20000   --save_iters_freq 20000  --optim 'Adam'  --niter 200000  --ker_size 7 \
#--checkpoints_dir '/apdcephfs/private_jiaxianchen/PISE_out/tkin_bone_k7' 
python3 -m torch.distributed.launch --nproc_per_node=4  train.py --name=fashion --model=boneline --dist=1  \
--batchSize=16 --nThreads=16  --lr 1e-4  --print_freq 100  --epochs 200  \
--display_freq=500 --use_rec=0 --point_line=1 \
--continue_train --save_latest_freq  10000   --save_iters_freq 10000  --optim 'Adam'  --niter 200000  --ker_size 7 \
--checkpoints_dir '/mnt/private_jiaxianchen/PISE_out/tkin_bone_k7' >> /mnt/private_jiaxianchen/PISE_out/tkin_bone_k7/out.txt 


python3 -m torch.distributed.launch --nproc_per_node=4  train.py --name=fashion --model=boneline --dist=1  \
--batchSize=16 --nThreads=16  --lr 1e-4  --print_freq 100  --epochs 200  \
--display_freq=500 --use_rec=0 --point_line=1 \
--continue_train --save_latest_freq  10000   --save_iters_freq 10000  --optim 'Adam'  --niter 200000  --ker_size 1 \
--checkpoints_dir '/mnt/private_jiaxianchen/PISE_out/tkin_bone_k1' >> /mnt/private_jiaxianchen/PISE_out/tkin_bone_k1/out.txt 