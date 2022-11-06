#python3 -m torch.distributed.launch --nproc_per_node=4  train.py --name=fashion --model=pure --dist=1  \
#--batchSize=16 --nThreads=16  --lr 1e-4  --print_freq 100  --epochs 200  \
#--display_freq=500 --sn=0 --gt_par=1  --use_rec=1 --use_shape=1 --patch_dec=1 \
#--continue_train --save_latest_freq  20000   --save_iters_freq 20000  --optim 'Adam'  --niter 200000  --ker_size 5 \
#--checkpoints_dir '/apdcephfs/private_jiaxianchen/PISE_out/10_11_patchdec' >> /apdcephfs/private_jiaxianchen/PISE_out/10_11_patchdec/out.txt   

#python3 -m torch.distributed.launch --nproc_per_node=4  train.py --name=fashion --model=pure --dist=1  \
#--batchSize=16 --nThreads=16  --lr 1e-4  --print_freq 100  --epochs 200  \
#--display_freq=500 --sn=0 --gt_par=1  --use_rec=1 --use_shape=1 \
#--continue_train --save_latest_freq  20000   --save_iters_freq 20000  --optim 'Adam'  --niter 200000  --ker_size 3 --iter_count 200000 \
#--checkpoints_dir '/apdcephfs/private_jiaxianchen/PISE_out/10_7_two_k3' >> /apdcephfs/private_jiaxianchen/PISE_out/10_7_two_k3/out.txt 

#python3 -m torch.distributed.launch --nproc_per_node=4  train.py --name=fashion --model=pure --dist=1  \
#--batchSize=16 --nThreads=16  --lr 1e-4  --print_freq 100  --epochs 200  \
#--display_freq=500 --sn=0 --gt_par=1  --use_rec=1 --use_shape=1 \
#--continue_train --save_latest_freq  20000   --save_iters_freq 20000  --optim 'Adam'  --niter 200000  --ker_size 7 --iter_count 200000 \
#--checkpoints_dir '/apdcephfs/private_jiaxianchen/PISE_out/10_7_two_k7' >> /apdcephfs/private_jiaxianchen/PISE_out/10_7_two_k7/out.txt 