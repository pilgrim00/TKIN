from options.test_options import TestOptions
import data as Dataset
from model import create_model
from util import visualizer
from itertools import islice
import numpy as np
import torch
import cv2 as cv
label_colors=[(0,0,0), (128,0,0), (255,0,0), (0,85,0), (170,0,51), (255,85,0), (0,0,85), (0,119,221)]
def paint_parsmap(SPL):
    #SPL:(1, 8, 256, 256),numpy
    out=np.zeros(shape=(256,256,3),dtype=np.uint8)
    for i in range(8):
        temp=SPL[0][i].reshape(256,256,1).repeat(3,axis=2)*label_colors[i]
        out+=temp.astype(np.uint8)
    #out=out[:,:,[2,1,0]]
    return out
if __name__=='__main__':
    # get testing options
    opt = TestOptions().parse(0)
    print(opt.phase)
    #exit()
    opt.serial_batches=True #shuffle=not opt.serial_batches,
    model = create_model(opt)
    # creat a dataset
    dataset = Dataset.create_dataloader(opt)

    dataset_size = len(dataset) * opt.batchSize
    print('testing images = %d' % dataset_size)
    datas=[]
    with torch.no_grad():
        for i, data in enumerate(dataset):
            datas.append(data)
            if i>20:
                break
        #------------------
        a=16
        model.set_input(datas[a])
        print(a,datas[a]['P1_path'],datas[a]['P2_path'])
        out=model.push_output()
        out=out.permute(1, 2, 0).cpu().numpy()
        out=(out+1.0)/2.0
        out=out[:,:,[2,1,0]]
        out=out*255.0
        out=out.astype(np.uint8)
        cv.imwrite('./puretransfer_dtd'+str(a)+'.png',out)
        SPL1=model.input_SPL1.cpu().numpy()
        SPL1=paint_parsmap(SPL1)
        SPL1=SPL1[:,:,[2,1,0]]
        cv.imwrite('./SPL1_'+str(a)+'.png',SPL1)
        SPL2=model.input_SPL2.cpu().numpy()
        SPL2=paint_parsmap(SPL2)
        SPL2=SPL2[:,:,[2,1,0]]
        cv.imwrite('./SPL2_'+str(a)+'.png',SPL2)
    