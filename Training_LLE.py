# -*- coding: utf-8 -*-
"""
@author: DongXiao
"""

import time, os
import torch
import datetime
from Convmixer_tau import ConvMixer_tau
from utils import load_data,paramter_initialize, train_model
try:
    from torchsummary import summary
except ImportError:
    summary = None
 #%%   
#-----------------------------------------------------------------------------#
if __name__ == '__main__':
    for i in range(1):
        dim = 16
        depth = 8
        ks = 9
        ps = 8
        
        
        USE_GPU = torch.cuda.is_available()
        Start = time.time()
        here = os.path.dirname(os.path.abspath(__file__))
        data_path = os.environ.get("FPFLI_TRAINING_DATA",
                                   os.path.join(here, "Training_data.mat"))
        path, data_name = os.path.split(data_path)
        path = path or "."
        now = datetime.datetime.now()
        
        train_set, test_set = load_data(path, data_name,BATCH_SIZE = 128)  
        log_root = os.environ.get("FPFLI_TRAINING_OUTPUT", here)
        ckpt_dir=os.path.join(log_root, 'training_logs_{}_{}_{}_{}_{}'.format(dim, depth, ks,ps,i))
        os.makedirs(ckpt_dir,exist_ok=True)
        model = ConvMixer_tau(dim,depth,ks,ps)
        if summary is not None:
            summary(model, [(1, 256),(1,256)],device='cpu')
        if USE_GPU:
            model.cuda() 
        paramter_initialize(model)
        Record = train_model(train_set,test_set, model,learning_rate = 1e-4,
                             Epoch=200, USE_GPU = USE_GPU, log_interval =200,
                             patience=20,lr_step = 20, lr_gamma = 0.8,
                             path_dir=ckpt_dir)    
    
        model_data = dict(
            model = model.state_dict(), Record=Record,
            info = 'trained with a maximum of 200 epochs and early stopping')
        torch.save(model_data, 
               os.path.join(ckpt_dir,'training.pth')) 
        Stop = time.time()-Start
        print('The training time is: ', Stop)
