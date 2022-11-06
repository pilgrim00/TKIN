#纹理迁移
python puretransfer_dtd.py  --checkpoints_dir  '/apdcephfs/private_jiaxianchen/PISE_out/11_12_bankk3'   --nThreads=0  \
--name=fashion --model=puretransferdtd   --dataset_mode dtdfashion  \
--use_rec=0 --ker_size 3 
