import argparse
import os
import torch
from exp.exp_retrieval import Exp_Retrieval
import random
import numpy as np
import json

parser = argparse.ArgumentParser("")
parser.add_argument("--model", type=str, default='Cogformer')
parser.add_argument("--seed", type=int, default=1, help="random seed")
parser.add_argument("--subject", type=str, default='subj01')
# data loader
parser.add_argument("--data", type=str, default="NSD", help="dataset type") 
# optimization
parser.add_argument("--num_workers", type=int, default=1, help="data loader num workers")
parser.add_argument("--activation", type=str, default="gelu", help="activation")
# GPU
parser.add_argument("--use_gpu", type=bool, default=True, help="use gpu")
parser.add_argument("--gpu", type=int, default=0, help="gpu")
parser.add_argument("--devices", type=str, default="0,1")
args = parser.parse_args()

args.task_name = 'retrieval'
args.root_path = 'dataset/NSD/processed_data/{}/'.format(args.subject)

fMRI_info = np.load(args.root_path + 'roi_vn.npz')
fMRI_info = {key: int(fMRI_info[key]) for key in fMRI_info}
args.roi_name = [key for key in fMRI_info.keys()]
args.batch_size = 128
args.brain_roi = 14
args.enc_in = min(list(fMRI_info.values()))
args.e_model = 256
args.d_ff = 512
args.n_heads = 8
args.e_layers = 3
args.hf_dim = 1024
args.lf_dim = 512
args.dropout = 0.3
args.learning_rate = 0.0001
args.train_epochs = 300
args.patience = 10
args.supercategories = 12
args.labels = 80
args.train_samples = 24980

print("Args in experiment:")
print(args)

if __name__ == "__main__":
    seed = args.seed
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    Exp = Exp_Retrieval
    exp = Exp(args)
    
    exp.train()
    # exp.results()
    # exp.save_feature()
    