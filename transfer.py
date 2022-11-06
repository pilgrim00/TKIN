from options.test_options import TestOptions
import data as Dataset
from model import create_model
from util import visualizer
from itertools import islice
import numpy as np
import torch
import cv2 as cv
if __name__=='__main__':
    # get testing options
    opt = TestOptions().parse(0)
    print(opt.phase)
    #exit()
    opt.serial_batches=False
    #opt.phase='train'
    opt.model='transfer'
    model = create_model(opt)
    # creat a dataset
    dataset = Dataset.create_dataloader(opt)

    dataset_size = len(dataset) * opt.batchSize
    print('testing images = %d' % dataset_size)
    datas=[]
    with torch.no_grad():
        for i, data in enumerate(dataset):
            datas.append(data)
            if i>100:
                break
    model.set_input(datas[50])
    model.save4trans(datas[80])
    out=model.transfer_tex()
    out=out.permute(1, 2, 0).cpu().numpy()
    out=(out+1.0)/2.0
    out1=out[:,:,[2,1,0]]
    #out2=out[:,:,[0,2,1]]
    #print(np.shape(out))
    cv.imwrite('./trans10.jpg',out1*255.0)
    #cv.imwrite('./trans2.jpg',out2*255.0)
    model.set_input(datas[80])
    model.save4trans(datas[50])
    out=model.transfer_tex()
    out=out.permute(1, 2, 0).cpu().numpy()
    out=(out+1.0)/2.0
    out1=out[:,:,[2,1,0]]
    #out2=out[:,:,[0,2,1]]
    #print(np.shape(out))
    cv.imwrite('./trans11.jpg',out1*255.0)
