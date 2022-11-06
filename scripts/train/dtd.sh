python3 -m torch.distributed.launch --nproc_per_node=4 train.py --name=dtdfashion --model=dtdpar --dist=1  --dataset_mode dtdfashion  \
--batchSize=8 --nThreads=8  --lr 1e-4  --print_freq 100  --epochs 200  \
--display_freq=500 --sn=0 --gt_par=1  --use_rec=1 --use_shape=1  --use_bank=0 --use_bank_org=0 \
--continue_train --save_latest_freq  20000   --save_iters_freq 20000  --optim 'Adam' --niter 250000 \
--checkpoints_dir '/apdcephfs/private_jiaxianchen/PISE_out/9_29_dtd' >> /apdcephfs/private_jiaxianchen/PISE_out/9_29_dtd/out.txt
exit
python3 -m torch.distributed.launch --nproc_per_node=4  train.py --name=dtdfashion --model=dtd --dist=1  --dataset_mode dtdfashion  \
--batchSize=8 --nThreads=8  --lr 1e-4  --print_freq 100  --epochs 200  \
--display_freq=500 --sn=0 --gt_par=1  --use_rec=1 --use_shape=1  --use_bank=0 --use_bank_org=0 \
--continue_train --save_latest_freq  20000   --save_iters_freq 20000  --optim 'Adam' --niter 1250000  \
--checkpoints_dir '/apdcephfs/private_jiaxianchen/PISE_out/9_27_dtd' >> /apdcephfs/private_jiaxianchen/PISE_out/9_27_dtd/out.txt
exit
python3 -m torch.distributed.launch --nproc_per_node=1  train.py --name=dtdfashion --model=dtd --dist=1  --dataset_mode dtdfashion  \
--batchSize=1 --nThreads=0  --lr 1e-4  --print_freq 10  --epochs 200  \
--display_freq=20 --sn=0 --gt_par=1  --use_rec=1 --use_shape=1  --use_bank=0 --use_bank_org=0 \
--continue_train --save_latest_freq  20000   --save_iters_freq 20000  --optim 'Adam' --niter 1250000  \
--checkpoints_dir '/apdcephfs/private_jiaxianchen/PISE_out/9_14_sean' >> /apdcephfs/private_jiaxianchen/PISE_out/9_14_sean/out.txt
exit
python3 -m torch.distributed.launch --nproc_per_node=1 train.py --name=dtdfashion --model=dtdpar --dist=1  --dataset_mode dtdfashion  \
--batchSize=1 --nThreads=0  --lr 1e-4  --print_freq 10  --epochs 200  \
--display_freq=50 --sn=0 --gt_par=1  --use_rec=1 --use_shape=1  --use_bank=0 --use_bank_org=0 \
--continue_train --save_latest_freq  20000   --save_iters_freq 20000  --optim 'Adam' --niter 1250000