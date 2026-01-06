import os 
import torch
import torch.nn as nn
from models import Cogformer

class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        self.model_dict = {
            "Cogformer": Cogformer,
            
        }
        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)

    def _build_model(self):
        raise NotImplementedError
        return None

    def _acquire_device(self):
        if self.args.use_gpu:
            device = torch.device("cuda:{}".format(self.args.gpu))
            print("Use GPU: cuda:{}".format(self.args.gpu))
        else:
            device = torch.device("cpu")
            print("Use CPU")
        return device