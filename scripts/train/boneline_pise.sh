python3 -m torch.distributed.launch --nproc_per_node=4  train.py --name=fashion --model=boneline --dist=1  \
--batchSize=16 --nThreads=16  --lr 1e-4  --print_freq 100  --epochs 200  \
--display_freq=200 --point_line=1 --par_weight=1 --parnet_pise=1  \
--continue_train --save_latest_freq  20000   --save_iters_freq 20000  --optim 'Adam'  --niter 200000  --ker_size 3 \
--checkpoints_dir '/apdcephfs/private_jiaxianchen/PISE_out/12_7_boneline_pise' >> /apdcephfs/private_jiaxianchen/PISE_out/12_7_boneline_pise/out.txt 
exit