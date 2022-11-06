#纹理迁移
python puretransfer.py  --checkpoints_dir  '/remote-home/pilgrim02/PISE_out/10_7_two_k3'   --nThreads=0 --name=fashion --model=puretransfer  \
--use_rec=0 --ker_size 3 
exit
#python puretransfer.py  --checkpoints_dir  '/apdcephfs/private_jiaxianchen/PISE_out/10_7_two_k7'   --nThreads=0 --name=fashion --model=puretransfer  \
#--use_rec=0 --ker_size 7
exit
#姿态迁移，保存姿态图
python pttest.py --checkpoints_dir '/apdcephfs/private_jiaxianchen/PISE_out/11_2_bonelineth_k5_bank' --nThreads=0 --name=fashion --model=bonelinethre  \
--results_dir  '/apdcephfs/private_jiaxianchen/PISE_out/11_2_bonelineth_k5_bank/pteval'  --use_rec=0 --ker_size 5 --point_line=1 --par_weight=1  --use_bank=1

python pttest.py --checkpoints_dir '/apdcephfs/private_jiaxianchen/PISE_out/11_2_bonelineth_k5' --nThreads=0 --name=fashion --model=bonelinethre  \
--results_dir  '/apdcephfs/private_jiaxianchen/PISE_out/11_2_bonelineth_k5/pteval'  --use_rec=0 --ker_size 5 --point_line=1 --par_weight=1  --use_bank=0