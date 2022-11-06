
python3 -m torch.distributed.launch --nproc_per_node=4  train.py --name=fashion --model=bonelinethre --dist=1  \
--batchSize=16 --nThreads=16  --lr 1e-4  --print_freq 100  --epochs 200  \
--display_freq=200 --use_rec=0 --point_line=1 \
--continue_train --save_latest_freq  20000   --save_iters_freq 20000  --optim 'Adam'  --niter 200000  --ker_size 3 \
--checkpoints_dir '/apdcephfs/private_jiaxianchen/PISE_out/10_22_bonelinethre' >> /apdcephfs/private_jiaxianchen/PISE_out/10_22_bonelinethre/out.txt 
exit
python3 -m torch.distributed.launch --nproc_per_node=1  train.py --name=fashion --model=bonelinethre --dist=1  \
--batchSize=1 --nThreads=0  --lr 1e-4  --print_freq 100  --epochs 200  \
--display_freq=200 --use_rec=0 --point_line=1 \
--continue_train --save_latest_freq  20000   --save_iters_freq 20000  --optim 'Adam'  --niter 200000  --ker_size 3 \
--checkpoints_dir '/apdcephfs/private_jiaxianchen/PISE_out/10_22_bonelinethre'
exit
python3 -m torch.distributed.launch --nproc_per_node=4  train.py --name=fashion --model=boneline --dist=1  \
--batchSize=16 --nThreads=16  --lr 1e-4  --print_freq 100  --epochs 200  \
--display_freq=200 --use_rec=0 --point_line=1 \
--continue_train --save_latest_freq  20000   --save_iters_freq 20000  --optim 'Adam'  --niter 200000  --ker_size 3 \
--checkpoints_dir '/apdcephfs/private_jiaxianchen/PISE_out/10_20_boneline' >> /apdcephfs/private_jiaxianchen/PISE_out/10_20_boneline/out.txt 
exit
python3 -m torch.distributed.launch --nproc_per_node=1  train.py --name=fashion --model=boneline --dist=1  \
--batchSize=1 --nThreads=0  --lr 1e-4  --print_freq 100  --epochs 200  \
--display_freq=100 --use_rec=0 --point_line=1 \
--continue_train --save_latest_freq  20000   --save_iters_freq 20000  --optim 'Adam'  --niter 200000  --ker_size 3 --iter_count 200000 \
exit