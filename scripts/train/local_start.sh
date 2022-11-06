#python train.py --name=fashion --model=painet --gpu_ids=0  --batchSize=1  --nThreads=0  \
#--checkpoints_dir  '/apdcephfs/private_jiaxianchen/PISE_out/8_23_patn_wosn'   \
#--sn=0 --gt_par=0  --use_shape=0  --lr 1e-6 --display_freq=1  --continue_train  \
#--lr 1e-6
python3 -m torch.distributed.launch --nproc_per_node=1  train.py --name=fashion --model=pure --dist=1  \
--batchSize=1 --nThreads=0  --lr 1e-4  --print_freq 100  --epochs 200  \
--display_freq=100 \
--continue_train --save_latest_freq  20000   --save_iters_freq 20000  --optim 'Adam'  --niter 200000  --ker_size 1 --iter_count 200000 \
--checkpoints_dir '/apdcephfs/private_jiaxianchen/PISE_out/tkin_k1' >> /apdcephfs/private_jiaxianchen/PISE_out/tkin_k1/out.txt 
exit
python3 -m torch.distributed.launch --nproc_per_node=1  train.py --name=fashion --model=texloc --dist=1  \
--batchSize=1 --nThreads=0  --lr 1e-4  --print_freq 200  --epochs 200  \
--display_freq=500 --sn=0 --gt_par=1  --use_rec=1 --use_shape=1  --use_bank=0 --use_bank_org=0 \
--continue_train --save_latest_freq  20000   --save_iters_freq 20000  --optim 'Adam'  
exit
python3 -m torch.distributed.launch --nproc_per_node=1  train.py --name=fashion --model=pure --dist=1  \
--batchSize=16 --nThreads=16  --lr 1e-4  --print_freq 20  --epochs 200  \
--display_freq=100 --sn=0 --gt_par=1  --use_rec=1 --use_shape=1  --use_bank=0 --use_bank_org=0 \
--continue_train --save_latest_freq  20000   --save_iters_freq 20000  --optim 'Adam'  --niter 200000 \ 
--checkpoints_dir '/apdcephfs/private_jiaxianchen/PISE_out/10_3_two' >> /apdcephfs/private_jiaxianchen/PISE_out/10_3_two/out.txt 
exit
python3 -m torch.distributed.launch --nproc_per_node=4  train.py --name=fashion --model=painetnvl --dist=1  \
--batchSize=16 --nThreads=16  --lr 1e-4  --print_freq 200  --epochs 200  \
--display_freq=500 --sn=0 --gt_par=1  --use_rec=1 --use_shape=1  --use_bank=0 --use_bank_org=0 \
--continue_train --save_latest_freq  20000   --save_iters_freq 20000  --optim 'Adam'  \
--checkpoints_dir '/apdcephfs/private_jiaxianchen/PISE_out/9_14_nvl' >> /apdcephfs/private_jiaxianchen/PISE_out/9_14_nvl/out.txt
exit
python3 -m torch.distributed.launch --nproc_per_node=1  train.py --name=fashion --model=painet --dist=1  \
--batchSize=2 --nThreads=0  --lr 1e-4  --print_freq 200  --epochs 100  \
--display_freq=500 --sn=0 --gt_par=1  --use_rec=1 --use_shape=1  --use_bank=1  \
--optim 'Adam'
python3   train.py --name=fashion --model=painet --dist=0 --dp=0 \
--batchSize=1 --nThreads=0  --lr 1e-4  --print_freq 200  --epochs 100  \
--display_freq=500 --sn=0 --gt_par=1  --use_rec=0 --use_shape=1  --use_bank=0 --use_bank_org=1 \
--print_freq 100
--checkpoints_dir '/apdcephfs/private_jiaxianchen/PISE_out/8_26_shape_bank' --continue_train --which_iter latest \
--save_latest_freq  20000   --save_iters_freq 20000  --optim 'Adam' >> /apdcephfs/private_jiaxianchen/PISE_out/8_26_shape_bank/out.txt
