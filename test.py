from options.test_options import TestOptions
import data as Dataset
from model import create_model
from util import visualizer
from itertools import islice
import numpy as np
import torch
import random
import os
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

    dataset_size = len(dataset) * opt.batchSize
    print('testing images = %d' % dataset_size)
    # create a model
    #model = create_model(opt)
    iou=0
    #if hasattr(model,'par_metric'):
    #    iou=1
    with torch.no_grad():
        for i, data in enumerate(dataset):
            model.set_input(data)
            #model.test_dtd()
            model.test()
            #model.get_pic()
            #if iou==1:
                #class_iou,miou=model.par_metric.value()
                #print(class_iou,miou)