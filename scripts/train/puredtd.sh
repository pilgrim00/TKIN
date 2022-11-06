#from scratch
python3 -m torch.distributed.launch --nproc_per_node=4  train.py --name=fashion --model=puredtd --dist=1  \
--dataset_mode dtdfashion  \
--batchSize=16 --nThreads=16  --lr 1e-4  --print_freq 100  --epochs 200  \
--display_freq=500 \
--continue_train --save_latest_freq  20000   --save_iters_freq 20000  --optim 'Adam'  --niter 200000  --ker_size 3 \
--checkpoints_dir '/mnt/private_jiaxianchen/PISE_out/12_10_puredtd_from_scratch' >> /mnt/private_jiaxianchen/PISE_out/12_10_puredtd_from_scratch/out.txt 
exit
python3 -m torch.distributed.launch --nproc_per_node=1  train.py --name=fashion --model=puredtd --dist=1  \
--dataset_mode dtdfashion  \
--batchSize=1 --nThreads=0  --lr 1e-4  --print_freq 100  --epochs 200  \
--display_freq=500 \
--continue_train --save_latest_freq  20000   --save_iters_freq 20000  --optim 'Adam'  --niter 200000  --ker_size 3 \
--checkpoints_dir '/mnt/private_jiaxianchen/PISE_out/12_10_puredtd_from_scratch' 
exit
python3 -m torch.distributed.launch --nproc_per_node=4  train.py --name=fashion --model=puredtd --dist=1  \
--dataset_mode dtdfashion  \
--batchSize=16 --nThreads=16  --lr 1e-4  --print_freq 100  --epochs 200  \
--display_freq=500 \
--continue_train --save_latest_freq  20000   --save_iters_freq 20000  --optim 'Adam'  --niter 200000  --ker_size 3 \
--checkpoints_dir '/mnt/private_jiaxianchen/PISE_out/12_10_puredtd' >> /mnt/private_jiaxianchen/PISE_out/12_10_puredtd/out.txt 
exit
python3 -m torch.distributed.launch --nproc_per_node=1  train.py --name=fashion --model=puredtd --dist=1  \
--dataset_mode dtdfashion  \
--batchSize=1 --nThreads=0  --lr 1e-4  --print_freq 100  --epochs 200  \
--display_freq=100 \
--continue_train --save_latest_freq  200   --save_iters_freq 20000  --optim 'Adam'  --niter 200000  --ker_size 3 \
--checkpoints_dir '/apdcephfs/private_jiaxianchen/PISE_out/12_10_puredtd'  \
--which_iter 20000
exit