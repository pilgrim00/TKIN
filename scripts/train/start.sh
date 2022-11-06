#!/bin/bash
#sleep 100h
#sh tkin_boneline.sh
#exit
sh tkin_test.sh
exit
sh pure_buchong.sh
exit
#sh puresw.sh
#exit
#sh puredtdmix.sh
#exit
#sh puredtd.sh
#exit
#sh puregmm_woup.sh
#exit
#sh boneline_pise.sh
#exit
#sh new_test.sh 
#exit
#sh puregmm.sh
#exit
#sh limbs_one_pise.sh
#exit
#sh limbs_pise.sh
#exit
#sh limbs_one.sh
#exit
#sh limbs.sh
#exit
#sh pres_test.sh
#exit
#sh puretransfer.sh 
#exit
#sh ./setup.sh
#sh boneline_k5.sh
#exit
#sh boneline_k5_bank.sh
#exit
#sh bonelineth_k5.sh
#exit
#sh bonelineth_k5_bank.sh
#exit
#sh purebankdtd.sh
#exit
#sh purebankk5dtd.sh
#exit
#sh purebankk7dtd.sh
#exit
#sh purebank.sh
#exit
#sh purebankk7.sh
#exit
#制作了新的镜像，不需要这样操作了。
#sh se.sh 
#exit
#sh purecx.sh
#exit
#sh purebankk5.sh
#exit
#sh featconcat.sh
#exit
#sh test.sh
#exit
#sleep 100h
#sh multi_pure_1.sh
#exit
#sh multi_pure.sh
#sh sean.sh
#sh texloc.sh
#sh newpar_weight.sh
#exit
#sh boneline_weight.sh
#exit
sh boneline.sh
exit
#sh two_start.sh
#sh dtd.sh
exit
#sh test.sh
#exit
#sh painet_nvl.sh
#exit
python3 -m torch.distributed.launch --nproc_per_node=4  train.py --name=fashion --model=painet --dist=1  \
--batchSize=16 --nThreads=16  --lr 1e-4  --print_freq 200  --epochs 200  \
--display_freq=500 --sn=0 --gt_par=1  --use_rec=0 --use_shape=1  --use_bank=0 --use_bank_org=1 \
--checkpoints_dir '/apdcephfs/private_jiaxianchen/PISE_out/9_10_bank_org'  --continue_train \
--save_latest_freq  20000   --save_iters_freq 20000  --optim 'Adam' >> /apdcephfs/private_jiaxianchen/PISE_out/9_10_bank_org/out.txt

#sh test.sh
#sleep 2h
#python train.py --name=fashion --model=painet --gpu_ids=0  --batchSize=16 --nThreads=16  \
#--display_freq=500   --checkpoints_dir '/apdcephfs/private_jiaxianchen/PISE_out/8_23_patn_wosn'  \
#--lr 1e-4  --print_freq 200  --epochs 100
