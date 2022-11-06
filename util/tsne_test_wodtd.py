from options.test_options import TestOptions
import data as Dataset
from model import create_model
from util import visualizer
from itertools import islice
import numpy as np
import torch
import random
import os
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import cv2 as cv
def Init_Seed(arg):
    '''
    Disable cudnn to maximize reproducibility
    '''
    # torch.cuda.cudnn_enabled = False
    random.seed(arg.seed)
    np.random.seed(arg.seed)
    torch.manual_seed(arg.seed)
    torch.cuda.manual_seed(arg.seed)
    torch.cuda.manual_seed_all(arg.seed)
    os.environ['PYTHONHASHSEED'] = str(arg.seed)
def norm(x):
    #normalize
    x=(x-np.min(x,0))/(np.max(x,0)-np.min(x,0))
    return x
def remove_outliers(x,th):
    num,_=x.shape
    #we think the outliers data lies in the DTD dataset
    mean=np.mean(x,axis=0)
    std=np.std(x)
    for i in range(num):
        diff=np.square(x[i]-mean).mean()
        if diff>th*std:
            x[i]=mean
    return x


if __name__=='__main__':
    # get testing options
    opt = TestOptions().parse(0)
    print(opt.phase)
    #Init_Seed(opt)
    dataset = Dataset.create_dataloader(opt)
    model = create_model(opt)
    #exit()
    # creat a dataset
    #dataset = Dataset.create_dataloader(opt)
    tsne = TSNE(n_components=2,init='pca')#random
    dataset_size = len(dataset) * opt.batchSize
    print('testing images = %d' % dataset_size)
    # create a model
    #model = create_model(opt)
    choose_num=500
    styles=torch.zeros(choose_num*3,256,1,1)
    with torch.no_grad():
        for i, data in enumerate(dataset):
            print(i)
            if i>=choose_num:
                break
            model.set_input(data)
            model.test_dtd()
            if i==0:
                person_gen=model.img_gen[0].permute(1, 2, 0).cpu().numpy()
                person_gen=(person_gen+1.0)/2
                person_gen*=255
                person_gen=person_gen.astype(np.uint8)
                cv.imwrite('./TSNE/img6_5.png',person_gen[:,:,[2,1,0]])
            temp_0=model.style_kernels[0][0]
            temp_3=model.style_kernels[0][3]
            temp_5=model.style_kernels[0][5]
            styles[i]=temp_0
            styles[i+choose_num*1]=temp_3
            styles[i+choose_num*2]=temp_5
    x=styles.reshape(choose_num*3,-1)
    result=tsne.fit_transform(x)
    result[:choose_num*1]=remove_outliers(result[:choose_num*1],100)
    result[choose_num*1:choose_num*2]=remove_outliers(result[choose_num*1:choose_num*2],100)
    result[choose_num*2:choose_num*3]=remove_outliers(result[choose_num*2:choose_num*3],100)
    result=norm(result)
    print(x.shape,result.shape)
    #deepfashion_3
    plt.xticks([])
    plt.yticks([])
    #plt.scatter(result[:choose_num*1,0], result[:choose_num*1,1],label = '$background$',c='b',marker='o')#0
    #plt.legend()
    plt.scatter(result[choose_num*1:choose_num*2,0], result[choose_num*1:choose_num*2,1],c='r',marker='o')#3
    #plt.legend()
    plt.scatter(result[choose_num*2:choose_num*3,0], result[choose_num*2:choose_num*3,1],c='g',marker='o')#5
    #plt.legend()
    plt.show()
    plt.savefig("./TSNE/stylemix_pca_wodtd_tkink1"+str(choose_num)+".png")