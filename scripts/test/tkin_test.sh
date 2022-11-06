#k1
#python test.py --checkpoints_dir '/mnt/private_jiaxianchen/PISE_out/tkin_k1' --nThreads=0 --name=fashion --model=pure  \
#--results_dir  '/mnt/private_jiaxianchen/PISE_out/tkin_k1/eval'  --ker_size 1  \
#--which_iter latest
#python ssim.py --path '/mnt/private_jiaxianchen/PISE_out/tkin_k1/eval' >> /mnt/private_jiaxianchen/PISE/tkin_out.txt
#python fid_GFLA.py --path '/mnt/private_jiaxianchen/PISE_out/tkin_k1/eval' >> /mnt/private_jiaxianchen/PISE/tkin_out.txt
#python lpips_orig.py --path '/mnt/private_jiaxianchen/PISE_out/tkin_k1/eval' >> /mnt/private_jiaxianchen/PISE/tkin_out.txt
#k1+dtd
#python test.py --checkpoints_dir '/mnt/private_jiaxianchen/PISE_out/tkin_k1_dtd' --nThreads=0 --name=fashion --model=puredtd  \
#--dataset_mode dtdfashion  --results_dir  '/mnt/private_jiaxianchen/PISE_out/tkin_k1_dtd/eval'  --ker_size 1  \
#--which_iter latest
#python ssim.py --path '/mnt/private_jiaxianchen/PISE_out/tkin_k1_dtd/eval' >> /mnt/private_jiaxianchen/PISE/tkin_out.txt
#python fid_GFLA.py --path '/mnt/private_jiaxianchen/PISE_out/tkin_k1_dtd/eval' >> /mnt/private_jiaxianchen/PISE/tkin_out.txt
#python lpips_orig.py --path '/mnt/private_jiaxianchen/PISE_out/tkin_k1_dtd/eval' >> /mnt/private_jiaxianchen/PISE/tkin_out.txt
#k5+dtd
#python test.py --checkpoints_dir '/mnt/private_jiaxianchen/PISE_out/tkin_k5_dtd' --nThreads=0 --name=fashion --model=puredtd  \
#--dataset_mode dtdfashion  --results_dir  '/mnt/private_jiaxianchen/PISE_out/tkin_k5_dtd/eval'  --ker_size 5  \
#--which_iter latest
#python ssim.py --path '/mnt/private_jiaxianchen/PISE_out/tkin_k5_dtd/eval' >> /mnt/private_jiaxianchen/PISE/tkin_out.txt
#python fid_GFLA.py --path '/mnt/private_jiaxianchen/PISE_out/tkin_k5_dtd/eval' >> /mnt/private_jiaxianchen/PISE/tkin_out.txt
#python lpips_orig.py --path '/mnt/private_jiaxianchen/PISE_out/tkin_k5_dtd/eval' >> /mnt/private_jiaxianchen/PISE/tkin_out.txt
#k7+dtd
#python test.py --checkpoints_dir '/mnt/private_jiaxianchen/PISE_out/tkin_k7_dtd' --nThreads=0 --name=fashion --model=puredtd  \
#--dataset_mode dtdfashion  --results_dir  '/mnt/private_jiaxianchen/PISE_out/tkin_k7_dtd/eval'  --ker_size 7  \
#--which_iter latest
#python ssim.py --path '/mnt/private_jiaxianchen/PISE_out/tkin_k7_dtd/eval' >> /mnt/private_jiaxianchen/PISE/tkin_out.txt
#python fid_GFLA.py --path '/mnt/private_jiaxianchen/PISE_out/tkin_k7_dtd/eval' >> /mnt/private_jiaxianchen/PISE/tkin_out.txt
#python lpips_orig.py --path '/mnt/private_jiaxianchen/PISE_out/tkin_k7_dtd/eval' >> /mnt/private_jiaxianchen/PISE/tkin_out.txt
#tkin_bone_k7
python test.py --checkpoints_dir '/mnt/private_jiaxianchen/PISE_out/tkin_bone_k7' --nThreads=0 --name=fashion --model=boneline  \
--results_dir  '/mnt/private_jiaxianchen/PISE_out/tkin_bone_k7/eval'  --ker_size 7  --use_rec=0 --point_line=1  \
--which_iter latest
python ssim.py --path '/mnt/private_jiaxianchen/PISE_out/tkin_bone_k7/eval' >> /mnt/private_jiaxianchen/PISE/tkin_out.txt
python fid_GFLA.py --path '/mnt/private_jiaxianchen/PISE_out/tkin_bone_k7/eval' >> /mnt/private_jiaxianchen/PISE/tkin_out.txt
python lpips_orig.py --path '/mnt/private_jiaxianchen/PISE_out/tkin_bone_k7/eval' >> /mnt/private_jiaxianchen/PISE/tkin_out.txt
#exit
#tkin_bonedtd_k1，还是要和parsing训练结果对应起来才可。
python test.py --checkpoints_dir '/mnt/private_jiaxianchen/PISE_out/tkin_bonelinedtd_k1' --nThreads=0 --name=fashion --model=bonelinedtd  \
--dataset_mode dtdfashion  --results_dir  '/mnt/private_jiaxianchen/PISE_out/tkin_bonelinedtd_k1/eval'    \
--ker_size 1  --use_rec=0 --point_line=1  --which_iter latest
python ssim.py --path '/mnt/private_jiaxianchen/PISE_out/tkin_bonelinedtd_k1/eval' >> /mnt/private_jiaxianchen/PISE/tkin_out.txt
python fid_GFLA.py --path '/mnt/private_jiaxianchen/PISE_out/tkin_bonelinedtd_k1/eval' >> /mnt/private_jiaxianchen/PISE/tkin_out.txt
python lpips_orig.py --path '/mnt/private_jiaxianchen/PISE_out/tkin_bonelinedtd_k1/eval' >> /mnt/private_jiaxianchen/PISE/tkin_out.txt
#tkin_bonedtd_k5
python test.py --checkpoints_dir '/mnt/private_jiaxianchen/PISE_out/tkin_bonelinedtd_k5' --nThreads=0 --name=fashion --model=bonelinedtd  \
--dataset_mode dtdfashion  --results_dir  '/mnt/private_jiaxianchen/PISE_out/tkin_bonelinedtd_k5/eval'    \
--ker_size 5  --use_rec=0 --point_line=1  --which_iter latest
python ssim.py --path '/mnt/private_jiaxianchen/PISE_out/tkin_bonelinedtd_k5/eval' >> /mnt/private_jiaxianchen/PISE/tkin_out.txt
python fid_GFLA.py --path '/mnt/private_jiaxianchen/PISE_out/tkin_bonelinedtd_k5/eval' >> /mnt/private_jiaxianchen/PISE/tkin_out.txt
python lpips_orig.py --path '/mnt/private_jiaxianchen/PISE_out/tkin_bonelinedtd_k5/eval' >> /mnt/private_jiaxianchen/PISE/tkin_out.txt
#k7
python test.py --checkpoints_dir '/mnt/private_jiaxianchen/PISE_out/tkin_bonelinedtd_k7' --nThreads=0 --name=fashion --model=bonelinedtd  \
--dataset_mode dtdfashion  --results_dir  '/mnt/private_jiaxianchen/PISE_out/tkin_bonelinedtd_k7/eval'    \
--ker_size 7  --use_rec=0 --point_line=1  --which_iter latest
python ssim.py --path '/mnt/private_jiaxianchen/PISE_out/tkin_bonelinedtd_k7/eval' >> /mnt/private_jiaxianchen/PISE/tkin_out.txt
python fid_GFLA.py --path '/mnt/private_jiaxianchen/PISE_out/tkin_bonelinedtd_k7/eval' >> /mnt/private_jiaxianchen/PISE/tkin_out.txt
python lpips_orig.py --path '/mnt/private_jiaxianchen/PISE_out/tkin_bonelinedtd_k7/eval' >> /mnt/private_jiaxianchen/PISE/tkin_out.txt