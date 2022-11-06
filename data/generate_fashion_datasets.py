import os
import shutil
from PIL import Image
#root_path='/apdcephfs/share_1227775/BaseAI/workspace/jiaxianchen/inshop/PISE'
root_path='/remote-home/share/inshop'
IMG_EXTENSIONS = [
'.jpg', '.JPG', '.jpeg', '.JPEG',
'.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

def is_image_file(filename):
	return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def make_dataset(dir):
	images = []
	assert os.path.isdir(dir), '%s is not a valid directory' % dir
	new_root = root_path
	if not os.path.exists(new_root):
		os.mkdir(new_root)

	train_root = root_path+'/train'
	if not os.path.exists(train_root):
		os.mkdir(train_root)

	test_root = root_path+'/test'
	if not os.path.exists(test_root):
		os.mkdir(test_root)

	train_images = []
	train_f = open(root_path+'/train.lst', 'r')
	for lines in train_f:
		lines = lines.strip()
		if lines.endswith('.jpg'):
			train_images.append(lines)

	test_images = []
	test_f = open(root_path+'/test.lst', 'r')
	for lines in test_f:
		lines = lines.strip()
		if lines.endswith('.jpg'):
			test_images.append(lines)

	#print(train_images, test_images)
	# print(train_images, test_images)
	num=0
	for root, _, fnames in sorted(os.walk(dir)):
		for fname in fnames:
			if is_image_file(fname):
				num+=1
				print(num)
				path = os.path.join(root, fname)
				#print(path)
				#path1=path.split('/') 
				path1 = path.replace(root_path, 'fashion')
				path_names = path1.split('/') 
				# path_names[2] = path_names[2].replace('_', '')
				path_names[3] = path_names[3].replace('_', '')
				path_names[4] = path_names[4].split('_')[0] + "_" + "".join(path_names[4].split('_')[1:])
				path_names = "".join(path_names)

				new_path = path_names
				#img = Image.open(path)
				#imgcrop = img.crop((40, 0, 216, 256)),目的是想要图片变成长方形
				if new_path in train_images:
					out_path=os.path.join(train_root, path_names)
					shutil.copy(path,out_path)
					#img.save(os.path.join(train_root, path_names))
				elif new_path in test_images:
					#img.save(os.path.join(test_root, path_names))
					out_path=os.path.join(test_root, path_names)
					shutil.copy(path,out_path)
# 
def make_simplefiles(dir):
	images = []
	assert os.path.isdir(dir), '%s is not a valid directory' % dir
	new_root = root_path+'/simple_img'
	if not os.path.exists(new_root):
		os.mkdir(new_root)

	#print(train_images, test_images)
	# print(train_images, test_images)
	num=0
	for root, _, fnames in sorted(os.walk(dir)):
		for fname in fnames:
			if is_image_file(fname):
				num+=1
				print(num)
				path = os.path.join(root, fname)
				#print(path)
				#path1=path.split('/') 
				path1 = path.replace(root_path+'/img', 'fashion')
				path_names = path1.split('/') 
				# path_names[2] = path_names[2].replace('_', '')
				path_names[3] = path_names[3].replace('_', '')
				path_names[4] = path_names[4].split('_')[0] + "_" + "".join(path_names[4].split('_')[1:])
				path_names = "".join(path_names)

				new_path = path_names
				out_path=os.path.join(new_root, path_names)
				shutil.copy(path,out_path)
if __name__=='__main__':				
	make_dataset(root_path)
	#make_simplefiles(root_path+'/img')
