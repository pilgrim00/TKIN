python3 -m torch.distributed.launch --nproc_per_node=8  train.py --name=fashion --model=limbs --dist=1  \
--batchSize=16 --nThreads=16  --lr=1e-4  --print_freq 100  --epochs=200  \
--display_freq=500 --use_rec=0 --point_line=1 --use_bank=0 --par_weight=1 \
--continue_train --save_latest_freq=20000   --save_iters_freq=20000  --optim 'Adam'  --niter=200000  --ker_size=3 \
--checkpoints_dir '/apdcephfs/private_jiaxianchen/PISE_out/11_23_limbs' >> /apdcephfs/private_jiaxianchen/PISE_out/11_23_limbs/out.txt 
exit
python3 -m torch.distributed.launch --nproc_per_node=1  train.py --name=fashion --model=limbs --dist=1  \
--batchSize=1 --nThreads=0  --lr=1e-4  --print_freq 100  --epochs=200  \
--display_freq=500 --use_rec=0 --point_line=1 --use_bank=0 --par_weight=1 \
--continue_train --save_latest_freq=20000   --save_iters_freq=20000  --optim 'Adam'  --niter=200000  --ker_size=3
