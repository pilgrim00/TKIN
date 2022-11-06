#1.
#python test.py  --checkpoints_dir  '/apdcephfs/private_jiaxianchen/PISE_out/result_wgtpar'   --nThreads=0  --name=fashion  \
#--results_dir  '/apdcephfs/private_jiaxianchen/PISE_out/result_wgtpar/eval'  --sn=1 --gt_par=1 --use_shape=0 --use_rec=0  \
#python fid.py --crop=1 --path '/apdcephfs/private_jiaxianchen/PISE_out/result_wgtpar/eval' >> /apdcephfs/private_jiaxianchen/PISE/out.txt
#2.
#python test.py  --checkpoints_dir  '/apdcephfs/private_jiaxianchen/PISE_out/result_wosn'   --nThreads=0  --name=fashion  \
#--results_dir  '/apdcephfs/private_jiaxianchen/PISE_out/result_wosn/eval'  --sn=0 --gt_par=1  --use_shape=0 --use_rec=0
#python fid.py  --path '/apdcephfs/private_jiaxianchen/PISE_out/result_wosn/eval' >> /apdcephfs/private_jiaxianchen/PISE/out.txt
#3.
#python test.py  --checkpoints_dir  '/apdcephfs/private_jiaxianchen/PISE_out/8_23_patn_wosn'   --nThreads=0  --name=fashion  \
#--results_dir  '/apdcephfs/private_jiaxianchen/PISE_out/8_23_patn_wosn/eval'  --sn=0 --gt_par=0  --use_shape=0 --use_rec=0  \
#--use_bank=0 
#python fid.py  --path '/apdcephfs/private_jiaxianchen/PISE_out/8_23_patn_wosn/eval' >> /apdcephfs/private_jiaxianchen/PISE/out.txt
#4.
#python test.py  --checkpoints_dir  '/apdcephfs/private_jiaxianchen/PISE_out/8_26_shape'   --nThreads=0  --name=fashion  \
#--results_dir  '/apdcephfs/private_jiaxianchen/PISE_out/8_26_shape/eval' --sn=0 --gt_par=1  --use_rec=0 --use_shape=1 --use_bank=0 
#python fid.py  --path '/apdcephfs/private_jiaxianchen/PISE_out/8_26_shape/eval' >> /apdcephfs/private_jiaxianchen/PISE/out.txt
#5.
#python test.py  --checkpoints_dir  '/apdcephfs/private_jiaxianchen/PISE_out/8_26_shape_rec'   --nThreads=0  --name=fashion  \
#--results_dir  '/apdcephfs/private_jiaxianchen/PISE_out/8_26_shape_rec/eval' --sn=0 --gt_par=1  --use_rec=1 --use_shape=1 --use_bank=0 
#python fid.py  --path '/apdcephfs/private_jiaxianchen/PISE_out/8_26_shape_rec/eval' >> /apdcephfs/private_jiaxianchen/PISE/out.txt
#6.这个不是真的的Bank
#python test.py  --checkpoints_dir  '/apdcephfs/private_jiaxianchen/PISE_out/8_30_shape_rec_bank'   --nThreads=0  --name=fashion  \
#--results_dir  '/apdcephfs/private_jiaxianchen/PISE_out/8_30_shape_rec_bank/eval' --sn=0 --gt_par=1  --use_rec=1 --use_shape=0  --use_bank=0  
#python fid.py  --path '/apdcephfs/private_jiaxianchen/PISE_out/8_30_shape_rec_bank/eval' >> /apdcephfs/private_jiaxianchen/PISE/out.txt
#7.梯度方式存在问题，bank模块定义改了
#python test.py  --checkpoints_dir  '/apdcephfs/private_jiaxianchen/PISE_out/9_3_shape_rec_bank'   --nThreads=0  --name=fashion  \
#--results_dir  '/apdcephfs/private_jiaxianchen/PISE_out/9_3_shape_rec_bank/eval' --sn=0 --gt_par=1  --use_rec=1 --use_shape=1  --use_bank=1  
#python fid.py  --path '/apdcephfs/private_jiaxianchen/PISE_out/9_3_shape_rec_bank/eval' >> /apdcephfs/private_jiaxianchen/PISE/out.txt
#8.梯度正常传递。
#python test.py  --checkpoints_dir  '/apdcephfs/private_jiaxianchen/PISE_out/9_4_shape_rec_bank'   --nThreads=0  --name=fashion  \
#--results_dir  '/apdcephfs/private_jiaxianchen/PISE_out/9_4_shape_rec_bank/eval' --sn=0 --gt_par=1  --use_rec=1 --use_shape=1  --use_bank=1  \
#--which_iter 110000 --use_bank_org=0
#python fid.py  --path '/apdcephfs/private_jiaxianchen/PISE_out/9_4_shape_rec_bank/eval' >> /apdcephfs/private_jiaxianchen/PISE/out.txt
#9.用4的预训练模型加载bank
#python test.py  --checkpoints_dir  '/apdcephfs/private_jiaxianchen/PISE_out/8_26_shape_bank'   --nThreads=0  --name=fashion  \
#--results_dir  '/apdcephfs/private_jiaxianchen/PISE_out/8_26_shape_bank/eval' --sn=0 --gt_par=1  --use_rec=0 --use_shape=1 --use_bank=1 
#python fid.py  --path '/apdcephfs/private_jiaxianchen/PISE_out/8_26_shape_bank/eval' >> /apdcephfs/private_jiaxianchen/PISE/out.txt
#10.定义的正交bank
#python test.py  --checkpoints_dir  '/apdcephfs/private_jiaxianchen/PISE_out/9_10_bank_org'   --nThreads=0  --name=fashion  \
#--results_dir  '/apdcephfs/private_jiaxianchen/PISE_out/9_10_bank_org/eval' --sn=0 --gt_par=1  --use_rec=0 --use_shape=1 --use_bank=0 \
#--use_bank_org=1 #--which_iter 140000
#python fid.py  --path '/apdcephfs/private_jiaxianchen/PISE_out/9_10_bank_org/eval' >> /apdcephfs/private_jiaxianchen/PISE/out.txt
#10.设计的stylecode编码引入新的信息
#python test.py  --checkpoints_dir  '/apdcephfs/private_jiaxianchen/PISE_out/9_15_texloc'   --nThreads=0 --name=fashion --model=texloc  \
#--results_dir  '/apdcephfs/private_jiaxianchen/PISE_out/9_15_texloc/eval' --sn=0 --gt_par=1  --use_rec=1 --use_shape=1 --use_bank=0 \
#--use_bank_org=0 #--which_iter 140000
#python fid.py  --path '/apdcephfs/private_jiaxianchen/PISE_out/9_15_texloc/eval' >> /apdcephfs/private_jiaxianchen/PISE/out.txt
#11.取消vgg的featureloss
#python test.py  --checkpoints_dir  '/apdcephfs/private_jiaxianchen/PISE_out/9_14_nvl'   --nThreads=0 --name=fashion --model=painetnvl   \
#--results_dir  '/apdcephfs/private_jiaxianchen/PISE_out/9_14_nvl/eval' --sn=0 --gt_par=1  --use_rec=1 --use_shape=1 --use_bank=0 \
#--use_bank_org=0 #--which_iter 140000
#python fid.py  --path '/apdcephfs/private_jiaxianchen/PISE_out/9_14_nvl/eval' >> /apdcephfs/private_jiaxianchen/PISE/out.txt
#12.sean的结构
#python test.py  --checkpoints_dir  '/apdcephfs/private_jiaxianchen/PISE_out/9_14_sean'   --nThreads=0 --name=fashion --model=sean   \
#--results_dir  '/apdcephfs/private_jiaxianchen/PISE_out/9_14_sean/eval' --sn=0 --gt_par=1  --use_rec=1 --use_shape=1 --use_bank=0 \
#--use_bank_org=0 #--which_iter 140000
#python fid.py  --path '/apdcephfs/private_jiaxianchen/PISE_out/9_14_sean/eval' >> /apdcephfs/private_jiaxianchen/PISE/out.txt
#13  10的延续，+变为concat
#python test.py  --checkpoints_dir  '/apdcephfs/private_jiaxianchen/PISE_out/9_15_texloccon'   --nThreads=0 --name=fashion --model=texloccon  \
#--results_dir  '/apdcephfs/private_jiaxianchen/PISE_out/9_15_texloccon/eval' --sn=0 --gt_par=1  --use_rec=1 --use_shape=1 --use_bank=0 \
#--use_bank_org=0 #--which_iter 140000
#python fid.py  --path '/apdcephfs/private_jiaxianchen/PISE_out/9_15_texloccon/eval' >> /apdcephfs/private_jiaxianchen/PISE/out.txt
#14 9_29_dtd
#python test.py  --checkpoints_dir  '/apdcephfs/private_jiaxianchen/PISE_out/9_29_dtd'   --nThreads=0 --name=dtdfashion --model=dtdpar --dataset_mode dtdfashion  \
#--results_dir  '/apdcephfs/private_jiaxianchen/PISE_out/9_29_dtd/eval' --sn=0 --gt_par=1  --use_rec=1 --use_shape=1  --use_bank=0 --use_bank_org=0
#python fid.py  --path '/apdcephfs/private_jiaxianchen/PISE_out/9_29_dtd/eval' >> /apdcephfs/private_jiaxianchen/PISE/out.txt
#15 10_3_two 
'''
python test.py  --checkpoints_dir  '/apdcephfs/private_jiaxianchen/PISE_out/10_3_two'   --nThreads=0 --name=fashion --model=pure  \
--results_dir  '/apdcephfs/private_jiaxianchen/PISE_out/10_3_two/eval' --use_rec=1 --ker_size 5
#python ssim.py --path '/apdcephfs/private_jiaxianchen/PISE_out/10_3_two/eval' >> /apdcephfs/private_jiaxianchen/PISE/out.txt
python fid.py  --path '/apdcephfs/private_jiaxianchen/PISE_out/10_3_two/eval' >> /apdcephfs/private_jiaxianchen/PISE/out.txt
#16 10_7_two_k3
python test.py  --checkpoints_dir  '/remote-home/pilgrim02/PISE_out/10_7_two_k3'   --nThreads=0 --name=fashion --model=pure  \
--results_dir  '/remote-home/pilgrim02/PISE_out/10_7_two_k3/eval' --use_rec=0 --ker_size 3
#python ssim.py  --path '/apdcephfs/private_jiaxianchen/PISE_out/10_7_two_k3/eval' >> /apdcephfs/private_jiaxianchen/PISE/out.txt
python fid.py  --path '/apdcephfs/private_jiaxianchen/PISE_out/10_7_two_k3/eval' >> /apdcephfs/private_jiaxianchen/PISE/out.txt
#16 10_7_two_k7
python test.py  --checkpoints_dir  '/apdcephfs/private_jiaxianchen/PISE_out/10_7_two_k7'   --nThreads=0 --name=fashion --model=pure  \
--results_dir  '/apdcephfs/private_jiaxianchen/PISE_out/10_7_two_k7/eval' --use_rec=0 --ker_size 7
#python ssim.py --path '/apdcephfs/private_jiaxianchen/PISE_out/10_7_two_k7/eval' >> /apdcephfs/private_jiaxianchen/PISE/out.txt
python fid.py  --path '/apdcephfs/private_jiaxianchen/PISE_out/10_7_two_k7/eval' >> /apdcephfs/private_jiaxianchen/PISE/out.txt
#17 k5_patch
python test.py  --checkpoints_dir  '/apdcephfs/private_jiaxianchen/PISE_out/10_11_patchdec'   --nThreads=0 --name=fashion --model=pure  \
--results_dir  '/apdcephfs/private_jiaxianchen/PISE_out/10_11_patchdec/eval' --use_rec=0 --ker_size 5 --patch_dec=1
#python ssim.py --path '/apdcephfs/private_jiaxianchen/PISE_out/10_11_patchdec/eval' >> /apdcephfs/private_jiaxianchen/PISE/out.txt
python fid.py  --path '/apdcephfs/private_jiaxianchen/PISE_out/10_11_patchdec/eval' >> /apdcephfs/private_jiaxianchen/PISE/out.txt

echo '28_test_'>>/apdcephfs/private_jiaxianchen/PISE/out.txt
#18 multi
python test.py --checkpoints_dir '/apdcephfs/private_jiaxianchen/PISE_out/10_12_multi' --nThreads=0 --name=fashion --model=multi_pure  \
--results_dir  '/apdcephfs/private_jiaxianchen/PISE_out/10_12_multi/eval'  --use_rec=0 --ker_size 3
python ssim.py --path '/apdcephfs/private_jiaxianchen/PISE_out/10_12_multi/eval' >> /apdcephfs/private_jiaxianchen/PISE/out.txt
python fid.py  --path '/apdcephfs/private_jiaxianchen/PISE_out/10_12_multi/eval' >> /apdcephfs/private_jiaxianchen/PISE/out.txt
python lpips_.py --path '/apdcephfs/private_jiaxianchen/PISE_out/10_12_multi/eval' >> /apdcephfs/private_jiaxianchen/PISE/out.txt
#19 multi_res
python test.py --checkpoints_dir '/apdcephfs/private_jiaxianchen/PISE_out/10_13_multi_1' --nThreads=0 --name=fashion --model=multi_pure_1  \
--results_dir  '/apdcephfs/private_jiaxianchen/PISE_out/10_13_multi_1/eval'  --use_rec=0 --ker_size 3 
python ssim.py --path '/apdcephfs/private_jiaxianchen/PISE_out/10_13_multi_1/eval' >> /apdcephfs/private_jiaxianchen/PISE/out.txt
python fid.py  --path '/apdcephfs/private_jiaxianchen/PISE_out/10_13_multi_1/eval' >> /apdcephfs/private_jiaxianchen/PISE/out.txt
python lpips_.py --path '/apdcephfs/private_jiaxianchen/PISE_out/10_13_multi_1/eval' >> /apdcephfs/private_jiaxianchen/PISE/out.txt
'''

#20 bone_line
#echo '22_boneline'>>/apdcephfs/private_jiaxianchen/PISE/out.txt
python test.py --checkpoints_dir '/remote-home/pilgrim02/PISE_out/10_20_boneline' --nThreads=4 --name=fashion --model=boneline  \
--results_dir  '/remote-home/pilgrim02/PISE_out/10_20_boneline/eval'  --use_rec=0 --ker_size 3 --point_line=1
#python test.py --checkpoints_dir '/apdcephfs/private_jiaxianchen/PISE_out/10_20_boneline' --nThreads=4 --name=fashion --model=boneline  \
#--results_dir  '/apdcephfs/private_jiaxianchen/PISE_out/10_20_boneline/eval'  --use_rec=0 --ker_size 3 --point_line=1
#python ssim.py  --path '/remote-home/pilgrim02/PISE_out/10_20_boneline/eval' >> /apdcephfs/private_jiaxianchen/PISE/out.txt
#python fid.py  --path '/remote-home/pilgrim02/PISE_out/10_20_boneline/eval' >> /apdcephfs/private_jiaxianchen/PISE/out.txt
#python fid_GFLA.py  --path '/remote-home/pilgrim02/PISE_out/10_20_boneline/eval' >> /apdcephfs/private_jiaxianchen/PISE/out.txt
#python lpips_.py --path '/remote-home/pilgrim02/PISE_out/10_20_boneline/eval' >> /apdcephfs/private_jiaxianchen/PISE/out.txt
#21 bone_line+softmax
#python test.py --checkpoints_dir '/apdcephfs/private_jiaxianchen/PISE_out/10_22_bonelinethre' --nThreads=4 --name=fashion --model=bonelinethre  \
#--results_dir  '/apdcephfs/private_jiaxianchen/PISE_out/10_22_bonelinethre/eval'  --use_rec=0 --ker_size 3 --point_line=1
#python ssim.py  --path '/apdcephfs/private_jiaxianchen/PISE_out/10_22_bonelinethre/eval' >> /apdcephfs/private_jiaxianchen/PISE/out.txt
#python fid.py  --path '/apdcephfs/private_jiaxianchen/PISE_out/10_22_bonelinethre/eval' >> /apdcephfs/private_jiaxianchen/PISE/out.txt
#python fid_GFLA.py  --path '/apdcephfs/private_jiaxianchen/PISE_out/10_22_bonelinethre/eval' >> /apdcephfs/private_jiaxianchen/PISE/out.txt
#python lpips_.py  --path '/apdcephfs/private_jiaxianchen/PISE_out/10_22_bonelinethre/eval' >> /apdcephfs/private_jiaxianchen/PISE/out.txt
#22 orignal_parnet+ourtexture
#python test.py --checkpoints_dir '/apdcephfs/private_jiaxianchen/PISE_out/10_25_newpar' --nThreads=0 --name=fashion --model=newpar  \
#--results_dir  '/apdcephfs/private_jiaxianchen/PISE_out/10_25_newpar/eval'  --use_rec=0 --ker_size 3 --point_line=0
#python ssim.py --path '/apdcephfs/private_jiaxianchen/PISE_out/10_25_newpar/eval' >> /apdcephfs/private_jiaxianchen/PISE/out.txt
#python fid.py --path '/apdcephfs/private_jiaxianchen/PISE_out/10_25_newpar/eval' >> /apdcephfs/private_jiaxianchen/PISE/out.txt
#python fid_GFLA.py --path '/apdcephfs/private_jiaxianchen/PISE_out/10_25_newpar/eval' >> /apdcephfs/private_jiaxianchen/PISE/out.txt
#python lpips_.py --path '/apdcephfs/private_jiaxianchen/PISE_out/10_25_newpar/eval' >> /apdcephfs/private_jiaxianchen/PISE/out.txt
#23  gt_parsmap+se_module 这个设置不太行
#python test.py --checkpoints_dir '/apdcephfs/private_jiaxianchen/PISE_out/10_25_se' --nThreads=0 --name=fashion --model=se  \
#--results_dir  '/apdcephfs/private_jiaxianchen/PISE_out/10_25_se/eval'  --use_rec=0 --ker_size 5 --point_line=0
#python ssim.py --path '/apdcephfs/private_jiaxianchen/PISE_out/10_25_se/eval' >> /apdcephfs/private_jiaxianchen/PISE/out.txt 
#python fid.py --path '/apdcephfs/private_jiaxianchen/PISE_out/10_25_se/eval' >> /apdcephfs/private_jiaxianchen/PISE/out.txt 
#python lpips_.py --path '/apdcephfs/private_jiaxianchen/PISE_out/10_25_se/eval' >> /apdcephfs/private_jiaxianchen/PISE/out.txt 
#24 bonline+weightpar
#python test.py --checkpoints_dir '/apdcephfs/private_jiaxianchen/PISE_out/10_27_boneline_weight' --nThreads=4 --name=fashion --model=boneline  \
#--results_dir  '/apdcephfs/private_jiaxianchen/PISE_out/10_27_boneline_weight/eval'  --use_rec=0 --ker_size 3 --point_line=1 --par_weight=1
#python ssim.py  --path '/apdcephfs/private_jiaxianchen/PISE_out/10_27_boneline_weight/eval' >> /apdcephfs/private_jiaxianchen/PISE/out.txt
#python fid.py  --path '/apdcephfs/private_jiaxianchen/PISE_out/10_27_boneline_weight/eval' >> /apdcephfs/private_jiaxianchen/PISE/out.txt
#python fid_GFLA.py  --path '/apdcephfs/private_jiaxianchen/PISE_out/10_27_boneline_weight/eval' >> /apdcephfs/private_jiaxianchen/PISE/out.txt
#python lpips_.py --path '/apdcephfs/private_jiaxianchen/PISE_out/10_27_boneline_weight/eval' >> /apdcephfs/private_jiaxianchen/PISE/out.txt
#25 newpar+weightpar
#python test.py --checkpoints_dir '/apdcephfs/private_jiaxianchen/PISE_out/10_27_newpar_weight' --nThreads=4 --name=fashion --model=newpar  \
#--results_dir  '/apdcephfs/private_jiaxianchen/PISE_out/10_27_newpar_weight/eval'  --use_rec=0 --ker_size 3 --point_line=0 --par_weight=1
#python ssim.py --path '/apdcephfs/private_jiaxianchen/PISE_out/10_27_newpar_weight/eval' >> /apdcephfs/private_jiaxianchen/PISE/out.txt
#python fid.py --path '/apdcephfs/private_jiaxianchen/PISE_out/10_27_newpar_weight/eval' >> /apdcephfs/private_jiaxianchen/PISE/out.txt
#python fid_GFLA.py --path '/apdcephfs/private_jiaxianchen/PISE_out/10_27_newpar_weight/eval' >> /apdcephfs/private_jiaxianchen/PISE/out.txt
#python lpips_.py --path '/apdcephfs/private_jiaxianchen/PISE_out/10_27_newpar_weight/eval' >> /apdcephfs/private_jiaxianchen/PISE/out.txt
#26feat_transfer送入 效果一般般
'''
python test.py --checkpoints_dir '/apdcephfs/private_jiaxianchen/PISE_out/10_28_featcat' --nThreads=4 --name=fashion --model=purefeatconcat  \
--results_dir  '/apdcephfs/private_jiaxianchen/PISE_out/10_28_featcat/eval'  --use_rec=0 --ker_size 7
python ssim.py --path '/apdcephfs/private_jiaxianchen/PISE_out/10_28_featcat/eval' >> /apdcephfs/private_jiaxianchen/PISE/out.txt
python fid.py --path '/apdcephfs/private_jiaxianchen/PISE_out/10_28_featcat/eval' >> /apdcephfs/private_jiaxianchen/PISE/out.txt
python fid_GFLA.py --path '/apdcephfs/private_jiaxianchen/PISE_out/10_28_featcat/eval' >> /apdcephfs/private_jiaxianchen/PISE/out.txt
python lpips_.py --path '/apdcephfs/private_jiaxianchen/PISE_out/10_28_featcat/eval' >> /apdcephfs/private_jiaxianchen/PISE/out.txt
'''
echo '11_15_test_again_again'>>/apdcephfs/private_jiaxianchen/PISE/out.txt
#27pure3*3filter+bank
#python test.py --checkpoints_dir '/apdcephfs/private_jiaxianchen/PISE_out/10_28_purebank' --nThreads=4 --name=fashion --model=purebank  \
#--results_dir  '/apdcephfs/private_jiaxianchen/PISE_out/10_28_purebank/eval'  --use_rec=0 --ker_size 3
#python ssim.py --path '/apdcephfs/private_jiaxianchen/PISE_out/10_28_purebank/eval' >> /apdcephfs/private_jiaxianchen/PISE/out.txt
#python fid.py --path '/apdcephfs/private_jiaxianchen/PISE_out/10_28_purebank/eval' >> /apdcephfs/private_jiaxianchen/PISE/out.txt
#python fid_GFLA.py --path '/apdcephfs/private_jiaxianchen/PISE_out/10_28_purebank/eval' >> /apdcephfs/private_jiaxianchen/PISE/out.txt
#python lpips_orig.py --path '/apdcephfs/private_jiaxianchen/PISE_out/10_28_purebank/eval' >> /apdcephfs/private_jiaxianchen/PISE/out.txt
#28pure5*5filter+bank
python test.py --checkpoints_dir '/apdcephfs/private_jiaxianchen/PISE_out/11_1_purebank' --nThreads=4 --name=fashion --model=purebank  \
--results_dir  '/apdcephfs/private_jiaxianchen/PISE_out/11_1_purebank/eval'  --use_rec=0 --ker_size 5
python ssim.py --path '/apdcephfs/private_jiaxianchen/PISE_out/11_1_purebank/eval' >> /apdcephfs/private_jiaxianchen/PISE/out.txt
#python fid.py --path '/apdcephfs/private_jiaxianchen/PISE_out/11_1_purebank/eval' >> /apdcephfs/private_jiaxianchen/PISE/out.txt
python fid_GFLA.py --path '/apdcephfs/private_jiaxianchen/PISE_out/11_1_purebank/eval' >> /apdcephfs/private_jiaxianchen/PISE/out.txt
python lpips_orig.py --path '/apdcephfs/private_jiaxianchen/PISE_out/11_1_purebank/eval' >> /apdcephfs/private_jiaxianchen/PISE/out.txt
#pure7*7filter+bank
#python test.py --checkpoints_dir '/apdcephfs/private_jiaxianchen/PISE_out/11_1_purebankk7' --nThreads=4 --name=fashion --model=purebank  \
#--results_dir  '/apdcephfs/private_jiaxianchen/PISE_out/11_1_purebankk7/eval'  --use_rec=0 --ker_size 7
#python ssim.py --path '/apdcephfs/private_jiaxianchen/PISE_out/11_1_purebankk7/eval' >> /apdcephfs/private_jiaxianchen/PISE/out.txt
#python fid.py --path '/apdcephfs/private_jiaxianchen/PISE_out/11_1_purebankk7/eval' >> /apdcephfs/private_jiaxianchen/PISE/out.txt
#python fid_GFLA.py --path '/apdcephfs/private_jiaxianchen/PISE_out/11_1_purebankk7/eval' >> /apdcephfs/private_jiaxianchen/PISE/out.txt
#python lpips_orig.py --path '/apdcephfs/private_jiaxianchen/PISE_out/11_1_purebankk7/eval' >> /apdcephfs/private_jiaxianchen/PISE/out.txt
#29pure5*5+sedecoder
#python test.py --checkpoints_dir '/apdcephfs/private_jiaxianchen/PISE_out/11_1_sede' --nThreads=4 --name=fashion --model=se  \
#--results_dir  '/apdcephfs/private_jiaxianchen/PISE_out/11_1_sede/eval'  --use_rec=0 --ker_size 5
#python ssim.py --path '/apdcephfs/private_jiaxianchen/PISE_out/11_1_sede/eval' >> /apdcephfs/private_jiaxianchen/PISE/out.txt
#python fid.py --path '/apdcephfs/private_jiaxianchen/PISE_out/11_1_sede/eval' >> /apdcephfs/private_jiaxianchen/PISE/out.txt
#python fid_GFLA.py --path '/apdcephfs/private_jiaxianchen/PISE_out/11_1_sede/eval' >> /apdcephfs/private_jiaxianchen/PISE/out.txt
#python lpips_.py --path '/apdcephfs/private_jiaxianchen/PISE_out/11_1_sede/eval' >> /apdcephfs/private_jiaxianchen/PISE/out.txt
#30pure5*5+cxloss
#python test.py --checkpoints_dir '/apdcephfs/private_jiaxianchen/PISE_out/11_1purecx' --nThreads=4 --name=fashion --model=purecx  \
#--results_dir  '/apdcephfs/private_jiaxianchen/PISE_out/11_1purecx/eval'  --use_rec=0 --ker_size 5
#python ssim.py --path '/apdcephfs/private_jiaxianchen/PISE_out/11_1purecx/eval' >> /apdcephfs/private_jiaxianchen/PISE/out.txt
#python fid.py --path '/apdcephfs/private_jiaxianchen/PISE_out/11_1purecx/eval' >> /apdcephfs/private_jiaxianchen/PISE/out.txt
#python fid_GFLA.py --path '/apdcephfs/private_jiaxianchen/PISE_out/11_1purecx/eval' >> /apdcephfs/private_jiaxianchen/PISE/out.txt
#python lpips_orig.py --path '/apdcephfs/private_jiaxianchen/PISE_out/11_1purecx/eval' >> /apdcephfs/private_jiaxianchen/PISE/out.txt
#31 boneline+thre+k5
#python test.py --checkpoints_dir '/apdcephfs/private_jiaxianchen/PISE_out/11_2_bonelineth_k5' --nThreads=4 --name=fashion --model=bonelinethre  \
#--results_dir  '/apdcephfs/private_jiaxianchen/PISE_out/11_2_bonelineth_k5/eval'  --use_rec=0 --ker_size 5 --point_line=1 --par_weight=1  --use_bank=0
#python ssim.py --path '/apdcephfs/private_jiaxianchen/PISE_out/11_2_bonelineth_k5/eval' >> /apdcephfs/private_jiaxianchen/PISE/out.txt
#python fid.py --path '/apdcephfs/private_jiaxianchen/PISE_out/11_2_bonelineth_k5/eval' >> /apdcephfs/private_jiaxianchen/PISE/out.txt
#python fid_GFLA.py --path '/apdcephfs/private_jiaxianchen/PISE_out/11_2_bonelineth_k5/eval' >> /apdcephfs/private_jiaxianchen/PISE/out.txt
#python lpips_orig.py --path '/apdcephfs/private_jiaxianchen/PISE_out/11_2_bonelineth_k5/eval' >> /apdcephfs/private_jiaxianchen/PISE/out.txt
#32 boneline+thre+k5+bank
#python test.py --checkpoints_dir '/apdcephfs/private_jiaxianchen/PISE_out/11_2_bonelineth_k5_bank' --nThreads=4 --name=fashion --model=bonelinethre  \
#--results_dir  '/apdcephfs/private_jiaxianchen/PISE_out/11_2_bonelineth_k5_bank/eval'  --use_rec=0 --ker_size 5 --point_line=1 --par_weight=1  --use_bank=1
#python ssim.py --path '/apdcephfs/private_jiaxianchen/PISE_out/11_2_bonelineth_k5_bank/eval' >> /apdcephfs/private_jiaxianchen/PISE/out.txt
#python fid.py --path '/apdcephfs/private_jiaxianchen/PISE_out/11_2_bonelineth_k5_bank/eval' >> /apdcephfs/private_jiaxianchen/PISE/out.txt
#python fid_GFLA.py --path '/apdcephfs/private_jiaxianchen/PISE_out/11_2_bonelineth_k5_bank/eval' >> /apdcephfs/private_jiaxianchen/PISE/out.txt
#python lpips_orig.py --path '/apdcephfs/private_jiaxianchen/PISE_out/11_2_bonelineth_k5_bank/eval' >> /apdcephfs/private_jiaxianchen/PISE/out.txt
#33 boneline+k5
#python test.py --checkpoints_dir '/apdcephfs/private_jiaxianchen/PISE_out/11_5_boneline_k5' --nThreads=4 --name=fashion --model=boneline  \
#--results_dir  '/apdcephfs/private_jiaxianchen/PISE_out/11_5_boneline_k5/eval'  --use_rec=0 --ker_size 5 --point_line=1 --par_weight=1  --use_bank=0
#python ssim.py --path '/apdcephfs/private_jiaxianchen/PISE_out/11_5_boneline_k5/eval' >> /apdcephfs/private_jiaxianchen/PISE/out.txt
#python fid.py --path '/apdcephfs/private_jiaxianchen/PISE_out/11_5_boneline_k5/eval' >> /apdcephfs/private_jiaxianchen/PISE/out.txt
#python fid_GFLA.py --path '/apdcephfs/private_jiaxianchen/PISE_out/11_5_boneline_k5/eval' >> /apdcephfs/private_jiaxianchen/PISE/out.txt
#python lpips_orig.py --path '/apdcephfs/private_jiaxianchen/PISE_out/11_5_boneline_k5/eval' >> /apdcephfs/private_jiaxianchen/PISE/out.txt
#34 boneline+k5+bank
#python test.py --checkpoints_dir '/apdcephfs/private_jiaxianchen/PISE_out/11_5_boneline_k5_bank' --nThreads=4 --name=fashion --model=boneline  \
#--results_dir  '/apdcephfs/private_jiaxianchen/PISE_out/11_5_boneline_k5_bank/eval'  --use_rec=0 --ker_size 5 --point_line=1 --par_weight=1  --use_bank=1
#python ssim.py --path '/apdcephfs/private_jiaxianchen/PISE_out/11_5_boneline_k5_bank/eval' >> /apdcephfs/private_jiaxianchen/PISE/out.txt
#python fid.py --path '/apdcephfs/private_jiaxianchen/PISE_out/11_5_boneline_k5_bank/eval' >> /apdcephfs/private_jiaxianchen/PISE/out.txt
#python fid_GFLA.py --path '/apdcephfs/private_jiaxianchen/PISE_out/11_5_boneline_k5_bank/eval' >> /apdcephfs/private_jiaxianchen/PISE/out.txt
#python lpips_orig.py --path '/apdcephfs/private_jiaxianchen/PISE_out/11_5_boneline_k5_bank/eval' >> /apdcephfs/private_jiaxianchen/PISE/out.txt
#35
#python test.py --checkpoints_dir '/apdcephfs/private_jiaxianchen/PISE_out/11_12_bankk3' --nThreads=0 --name=fashion --model=purebankdtd   --dataset_mode dtdfashion  \
#--results_dir  '/apdcephfs/private_jiaxianchen/PISE_out/11_12_bankk3/eval'  --use_rec=0 --ker_size 3
#python ssim.py --path '/apdcephfs/private_jiaxianchen/PISE_out/11_12_bankk3/eval' >> /apdcephfs/private_jiaxianchen/PISE/out.txt
#python fid_GFLA.py --path '/apdcephfs/private_jiaxianchen/PISE_out/11_12_bankk3/eval' >> /apdcephfs/private_jiaxianchen/PISE/out.txt
#python lpips_orig.py --path '/apdcephfs/private_jiaxianchen/PISE_out/11_12_bankk3/eval' >> /apdcephfs/private_jiaxianchen/PISE/out.txt
#36
python test.py --checkpoints_dir '/apdcephfs/private_jiaxianchen/PISE_out/11_12_bankk5' --nThreads=0 --name=fashion --model=purebankdtd   --dataset_mode dtdfashion  \
--results_dir  '/apdcephfs/private_jiaxianchen/PISE_out/11_12_bankk5/eval'  --use_rec=0 --ker_size 5
python ssim.py --path '/apdcephfs/private_jiaxianchen/PISE_out/11_12_bankk5/eval' >> /apdcephfs/private_jiaxianchen/PISE/out.txt
python fid_GFLA.py --path '/apdcephfs/private_jiaxianchen/PISE_out/11_12_bankk5/eval' >> /apdcephfs/private_jiaxianchen/PISE/out.txt
python lpips_orig.py --path '/apdcephfs/private_jiaxianchen/PISE_out/11_12_bankk5/eval' >> /apdcephfs/private_jiaxianchen/PISE/out.txt
#37
python test.py --checkpoints_dir '/apdcephfs/private_jiaxianchen/PISE_out/11_12_bankk7' --nThreads=0 --name=fashion --model=purebankdtd   --dataset_mode dtdfashion  \
--results_dir  '/apdcephfs/private_jiaxianchen/PISE_out/11_12_bankk7/eval'  --use_rec=0 --ker_size 7
python ssim.py --path '/apdcephfs/private_jiaxianchen/PISE_out/11_12_bankk7/eval' >> /apdcephfs/private_jiaxianchen/PISE/out.txt
python fid_GFLA.py --path '/apdcephfs/private_jiaxianchen/PISE_out/11_12_bankk7/eval' >> /apdcephfs/private_jiaxianchen/PISE/out.txt
python lpips_orig.py --path '/apdcephfs/private_jiaxianchen/PISE_out/11_12_bankk7/eval' >> /apdcephfs/private_jiaxianchen/PISE/out.txt