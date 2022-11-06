#!/bin/bash
#pure+sw_loss
#python test.py --checkpoints_dir '/mnt/private_jiaxianchen/PISE_out/12_30_pure_sw' --nThreads=0 --name=fashion --model=puresw  \
#--dataset_mode fashion  --results_dir  '/mnt/private_jiaxianchen/PISE_out/12_30_pure_sw/eval'  --ker_size 3  \
#--which_iter latest
#python ssim.py --path '/mnt/private_jiaxianchen/PISE_out/12_30_pure_sw/eval' >> /mnt/private_jiaxianchen/PISE/new_out.txt
#python fid_GFLA.py --path '/mnt/private_jiaxianchen/PISE_out/12_30_pure_sw/eval' >> /mnt/private_jiaxianchen/PISE/new_out.txt
#python lpips_orig.py --path '/mnt/private_jiaxianchen/PISE_out/12_30_pure_sw/eval' >> /mnt/private_jiaxianchen/PISE/new_out.txt
#exit
#puredtd+mix_training(D_constrains)
#python test.py --checkpoints_dir '/mnt/private_jiaxianchen/PISE_out/12_10_puredtdmix' --nThreads=0 --name=fashion --model=puredtdmix  \
#--dataset_mode dtdfashion  --results_dir  '/mnt/private_jiaxianchen/PISE_out/12_10_puredtdmix/eval'  --ker_size 3  \
#--which_iter latest
#python ssim.py --path '/mnt/private_jiaxianchen/PISE_out/12_10_puredtdmix/eval' >> /mnt/private_jiaxianchen/PISE/new_out.txt
#python fid_GFLA.py --path '/mnt/private_jiaxianchen/PISE_out/12_10_puredtdmix/eval' >> /mnt/private_jiaxianchen/PISE/new_out.txt
#python lpips_orig.py --path '/mnt/private_jiaxianchen/PISE_out/12_10_puredtdmix/eval' >> /mnt/private_jiaxianchen/PISE/new_out.txt
#exit
#puredtd_from_scratch
python test.py --checkpoints_dir '/mnt/private_jiaxianchen/PISE_out/12_10_puredtd_from_scratch' --nThreads=0 --name=fashion --model=puredtd  \
--dataset_mode dtdfashion  --results_dir  '/mnt/private_jiaxianchen/PISE_out/12_10_puredtd_from_scratch/eval'  --ker_size 3  \
--which_iter latest
python ssim.py --path '/mnt/private_jiaxianchen/PISE_out/12_10_puredtd_from_scratch/eval' >> /mnt/private_jiaxianchen/PISE/new_out.txt
python fid_GFLA.py --path '/mnt/private_jiaxianchen/PISE_out/12_10_puredtd_from_scratch/eval' >> /mnt/private_jiaxianchen/PISE/new_out.txt
python lpips_orig.py --path '/mnt/private_jiaxianchen/PISE_out/12_10_puredtd_from_scratch/eval' >> /mnt/private_jiaxianchen/PISE/new_out.txt
exit
#puredtd
python test.py --checkpoints_dir '/mnt/private_jiaxianchen/PISE_out/12_10_puredtd' --nThreads=0 --name=fashion --model=puredtd  \
--dataset_mode dtdfashion  --results_dir  '/mnt/private_jiaxianchen/PISE_out/12_10_puredtd/dtd_eval'  --ker_size 3  \
--which_iter 220000
python ssim.py --path '/mnt/private_jiaxianchen/PISE_out/12_10_puredtd/eval' >> /mnt/private_jiaxianchen/PISE/new_out.txt
python fid_GFLA.py --path '/mnt/private_jiaxianchen/PISE_out/12_10_puredtd/eval' >> /mnt/private_jiaxianchen/PISE/new_out.txt
python lpips_orig.py --path '/mnt/private_jiaxianchen/PISE_out/12_10_puredtd/eval' >> /mnt/private_jiaxianchen/PISE/new_out.txt
exit
python tsne_test.py --checkpoints_dir '/mnt/private_jiaxianchen/PISE_out/12_10_puredtd' --nThreads=0 --name=fashion --model=puredtd  \
--dataset_mode dtdfashion  --results_dir  '/mnt/private_jiaxianchen/PISE_out/12_10_puredtd/dtd_eval'  --ker_size 3  \
--which_iter 220000
#0.gmm_woup
#python test.py --checkpoints_dir '/apdcephfs/private_jiaxianchen/PISE_out/12_10_gmm_woup' --nThreads=0 --name=fashion --model=puregmm_woup  \
#--results_dir  '/apdcephfs/private_jiaxianchen/PISE_out/12_10_gmm_woup/eval'  --use_rec=0 --ker_size 3
#python ssim.py --path '/apdcephfs/private_jiaxianchen/PISE_out/12_10_gmm_woup/eval' >> /apdcephfs/private_jiaxianchen/PISE/new_out.txt
#python fid_GFLA.py --path '/apdcephfs/private_jiaxianchen/PISE_out/12_10_gmm_woup/eval' >> /apdcephfs/private_jiaxianchen/PISE/new_out.txt
#python lpips_orig.py --path '/apdcephfs/private_jiaxianchen/PISE_out/12_10_gmm_woup/eval' >> /apdcephfs/private_jiaxianchen/PISE/new_out.txt
#exit
#pise's encoding ways 
python tsne_pise_test.py  --checkpoints_dir  '/apdcephfs/private_jiaxianchen/PISE_out/8_26_shape'   --nThreads=0  --name=fashion  \
--results_dir  '/apdcephfs/private_jiaxianchen/PISE_out/8_26_shape/eval' --sn=0 --gt_par=1  --use_rec=0 --use_shape=1 --use_bank=0 
#our tkin 1*1 encoding
python tsne_test_wodtd.py  --checkpoints_dir  '/apdcephfs/private_jiaxianchen/PISE_out/tkin_k1'   --nThreads=0  --name=fashion  --model=pure  \
--use_rec=0 --ker_size 1
#1.gmm
#python test.py --checkpoints_dir '/apdcephfs/private_jiaxianchen/PISE_out/12_1_puregmm' --nThreads=0 --name=fashion --model=puregmm  \
#--results_dir  '/apdcephfs/private_jiaxianchen/PISE_out/12_1_puregmm/eval'  --use_rec=0 --ker_size 3
#python ssim.py --path '/apdcephfs/private_jiaxianchen/PISE_out/12_1_puregmm/eval' >> /apdcephfs/private_jiaxianchen/PISE/new_out.txt
#python fid_GFLA.py --path '/apdcephfs/private_jiaxianchen/PISE_out/12_1_puregmm/eval' >> /apdcephfs/private_jiaxianchen/PISE/new_out.txt
#python lpips_orig.py --path '/apdcephfs/private_jiaxianchen/PISE_out/12_1_puregmm/eval' >> /apdcephfs/private_jiaxianchen/PISE/new_out.txt
#exit
#2.limbs_our
python test.py --checkpoints_dir '/apdcephfs/private_jiaxianchen/PISE_out/11_23_limbs' --nThreads=0 --name=fashion --model=limbs  \
--results_dir  '/apdcephfs/private_jiaxianchen/PISE_out/11_23_limbs/peval'  --use_rec=0 --ker_size 3 --point_line=1  \
#3
#python test.py --checkpoints_dir '/apdcephfs/private_jiaxianchen/PISE_out/11_23_limbs_one' --nThreads=0 --name=fashion --model=limbs  \
#--results_dir  '/apdcephfs/private_jiaxianchen/PISE_out/11_23_limbs_one/peval'  --use_rec=0 --ker_size 3 --point_line=1  --limb_one=1 \
#4
python test.py --checkpoints_dir '/apdcephfs/private_jiaxianchen/PISE_out/12_1_limbs_pise' --nThreads=0 --name=fashion --model=limbs  \
--results_dir  '/apdcephfs/private_jiaxianchen/PISE_out/12_1_limbs_pise/peval'  --use_rec=0 --ker_size 3 --point_line=1  --parnet_pise=1  \
#5
#python test.py --checkpoints_dir '/apdcephfs/private_jiaxianchen/PISE_out/12_1_limbs_one_pise' --nThreads=0 --name=fashion --model=limbs  \
#--results_dir  '/apdcephfs/private_jiaxianchen/PISE_out/12_1_limbs_one_pise/peval'  --use_rec=0 --ker_size 3 --point_line=1  --limb_one=1 --parnet_pise=1  \
#--which_iter 160000
#6
python test.py --checkpoints_dir '/apdcephfs/private_jiaxianchen/PISE_out/10_20_boneline' --nThreads=0 --name=fashion --model=boneline  \
--results_dir  '/apdcephfs/private_jiaxianchen/PISE_out/10_20_boneline/peval'  --use_rec=0 --ker_size 3 --point_line=1  \
--which_iter 680000
#7
python test.py --checkpoints_dir '/apdcephfs/private_jiaxianchen/PISE_out/10_22_bonelinethre' --nThreads=0 --name=fashion --model=bonelinethre  \
--results_dir  '/apdcephfs/private_jiaxianchen/PISE_out/10_22_bonelinethre/peval'  --use_rec=0 --ker_size 3 --point_line=1
#8 boneline_pise
python test.py --checkpoints_dir '/apdcephfs/private_iaxianchen/PISE_out/12_7_boneline_pise' --nThreads=0 --name=fashion --model=boneline  \
--results_dir  '/apdcephfs/private_jiaxianchen/PISE_out/12_7_boneline_pise/peval'  --use_rec=0 --ker_size 3 --point_line=1  --parnet_pise=1 
# keypoints+pise
python test.py --checkpoints_dir '/apdcephfs/private_jiaxianchen/PISE_out/10_25_newpar' --nThreads=0 --name=fashion --model=newpar  \
--results_dir  '/apdcephfs/private_jiaxianchen/PISE_out/10_25_newpar/peval'  --use_rec=0 --ker_size 3 --point_line=0
python test.py --checkpoints_dir '/apdcephfs/private_jiaxianchen/PISE_out/10_27_newpar_weight' --nThreads=0 --name=fashion --model=newpar  \
--results_dir  '/apdcephfs/private_jiaxianchen/PISE_out/10_27_newpar_weight/peval'  --use_rec=0 --ker_size 3 --point_line=0
#test the mode
python tsne_test_wodtd.py  --checkpoints_dir  '/apdcephfs/private_jiaxianchen/PISE_out/10_7_two_k3'   --nThreads=0 --name=fashion --model=pure  \
--results_dir  '/apdcephfs/private_jiaxianchen/PISE_out/10_7_two_k3/eval' --use_rec=0 --ker_size 3  \
--which_iter 360000
#文章开头的效果图。
python test.py --checkpoints_dir '/apdcephfs/private_jiaxianchen/PISE_out/10_20_boneline' --nThreads=0 \
--name=fashion --model=boneline  --results_dir  '/apdcephfs/private_jiaxianchen/PISE_out/10_20_boneline/showeval'  \
--use_rec=0 --ker_size 3 --point_line=1  --which_iter 680000 