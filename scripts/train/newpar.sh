python3 -m torch.distributed.launch --nproc_per_node=1  train.py --name=fashion --model=newpar --dist=1  \
--batchSize=1 --nThreads=0  --lr 1e-4  --print_freq 100  --epochs 200  \
--display_freq=200 --use_rec=0 --point_line=0 --par_weight=0 \
--continue_train --save_latest_freq  20000   --save_iters_freq 20000  --optim 'Adam'  --niter 200000  --ker_size 3 --iter_count 0  \
--checkpoints_dir '/apdcephfs/private_jiaxianchen/PISE_out/10_25_newpar'
exit

python3 -m torch.distributed.launch --nproc_per_node=1  train.py --name=fashion --model=newpar --dist=1  \
--batchSize=8 --nThreads=16  --lr 1e-4  --print_freq 100  --epochs 200  \
--display_freq=200 --use_rec=0 --point_line=0 --par_weight=0  \
--continue_train --save_latest_freq  20000   --save_iters_freq 20000  --optim 'Adam'  --niter 200000  --ker_size 3 --iter_count 0 \
--checkpoints_dir '/apdcephfs/private_jiaxianchen/PISE_out/10_25_newpar' >> /apdcephfs/private_jiaxianchen/PISE_out/10_25_newpar/out.txt
exit
