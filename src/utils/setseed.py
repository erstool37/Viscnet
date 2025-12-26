import os
import random
import numpy as np
import torch

def set_seed(seed=42):
    random.seed(seed)                      
    np.random.seed(seed)                  
    torch.manual_seed(seed)                 
    torch.cuda.manual_seed(seed)              
    torch.cuda.manual_seed_all(seed)        
    os.environ['PYTHONHASHSEED'] = str(seed)  
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.mkldnn.deterministic = True
    torch.backends.mkldnn.benchmark = False