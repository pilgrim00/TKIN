import time
from options.train_options import TrainOptions
import data as Dataset
from model import create_model
from util.visualizer import Visualizer
import torch
import os
import torch
import numpy as np
import random
from torch import distributed as dist
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
def get_rank():
    if not dist.is_available():
        return 0

    if not dist.is_initialized():
        return 0

    return dist.get_rank()
def synchronize():
    if not dist.is_available():
        return

    if not dist.is_initialized():
        return

    world_size = dist.get_world_size()

    if world_size == 1:
        return

    dist.barrier()
if __name__ == '__main__':
    # get training options
    opt = TrainOptions().parse(get_rank())
    if opt.dist==1:
        #n_gpu = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
        #rank=int(os.environ['INDEX'])    
        torch.cuda.set_device(opt.local_rank)
        #ip=os.environ['CHIEF_IP']
        #host_addr = 'tcp://' + ip + ':' + str(30000)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        #torch.distributed.init_process_group(backend="nccl", init_method=host_addr,rank=opt.local_rank+4*rank,world_size=n_gpu)
        synchronize()
    # create a model
    #Init_Seed(opt)
    model = create_model(opt)
    #model.save_networks('latest')
    #exit()
    # create a dataset
    dataset = Dataset.create_dataloader(opt)
    dataset_size = len(dataset) * opt.batchSize
    print('training images = %d' % dataset_size)

    # model = model.to()  
    # create a visualizer
    visualizer = Visualizer(opt)
    # training flag
    keep_training = True
    max_iteration = opt.niter
    epoch = 0
    total_iteration = model.opt.iter_count

    # training process
    while(keep_training):
        epoch_start_time = time.time()
        epoch+=1
        print('\n Training epoch: %d' % epoch)

        for i, data in enumerate(dataset):
            iter_start_time = time.time()
            total_iteration += 1
            model.opt.iter_count+=1
            model.set_input(data)
            #with torch.autograd.set_detect_anomaly(False):
            model.optimize_parameters()

            # display images on visdom and save images
            if (opt.dist==1 and opt.local_rank==0) or opt.dist==0:
                if total_iteration % opt.display_freq == 0:
                    #显示结果的关键之处
                    visualizer.display_current_results(model.get_current_visuals(), epoch,total_iteration)
                    #if hasattr(model, 'distribution'):
                        #visualizer.plot_current_distribution(model.get_current_dis()) 

                # print training loss and save logging information to the disk
                if total_iteration % opt.print_freq == 0:
                    losses = model.get_current_errors()
                    t = (time.time() - iter_start_time) / opt.batchSize
                    visualizer.print_current_errors(epoch, total_iteration, losses, t)
                    if opt.display_id > 0:
                        visualizer.plot_current_errors(total_iteration, losses)

                #if total_iteration % opt.eval_iters_freq == 0:
                    #model.eval() 
                    #if hasattr(model, 'eval_metric_name'):
                        #eval_results = model.get_current_eval_results()  
                        #visualizer.print_current_eval(epoch, total_iteration, eval_results)
                        #if opt.display_id > 0:
                            #visualizer.plot_current_score(total_iteration, eval_results)
                # save the model every <save_iter_freq> iterations to the disk
                if total_iteration % opt.save_iters_freq == 0:
                    print('saving the model of iterations %d' % total_iteration)
                    model.save_networks(total_iteration)
                #if total_iteration > max_iteration:
                    #keep_training = False
                    #break
                # save the latest model every <save_latest_freq> iterations to the disk
                if total_iteration % opt.save_latest_freq == 0:
                    print('saving the latest model (epoch %d, total_steps %d)' % (epoch, total_iteration))
                    model.save_networks('latest')




        model.update_learning_rate()


    print('\nEnd training')