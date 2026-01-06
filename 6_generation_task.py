import argparse
import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from models.diffusion_prior import *
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

parser = argparse.ArgumentParser("")
parser.add_argument("--seed", type=int, default=1, help="random seed")
parser.add_argument("--subject", type=str, default='subj01')
parser.add_argument("--device", type=str, default="cuda:0")
args = parser.parse_args()

args.task_name = 'generation'
args.root_path = 'dataset/NSD/processed_data/'+args.subject

def train():
    print(">>>>>>>start training: {}, {}, seed{} >>>>>>>>>>>>>>>>>>>>>>>>>>".format(args.task_name, args.subject, args.seed))

    train_data = np.load(args.root_path + '/train_sample_diffprior.npz', allow_pickle=True)
    train_fMRI_hf = train_data['train_fMRI_hf'].astype(np.float32)
    train_fMRI_lf = train_data['train_fMRI_lf'].astype(np.float32)
    train_image_hf = train_data['train_image_hf'].astype(np.float32)
    train_image_lf = train_data['train_image_lf'].astype(np.float32)
    train_sample_index = train_data['train_sample_index']

    dataset = EmbeddingDataset(
        c_embeddings=train_fMRI_hf, h_embeddings=train_image_hf,
    )
    dl = DataLoader(dataset, batch_size=1024, shuffle=True, num_workers=1)

    diffusion_prior = DiffusionPriorUNet(embed_dim=1024, cond_dim=1024, dropout=0.1)

    print(sum(p.numel() for p in diffusion_prior.parameters() if p.requires_grad))
    pipe = Pipe(diffusion_prior, device=args.device)

    pipe.train(dl, num_epochs=150, learning_rate=1e-3)

    path = './checkpoints/diffusion_prior/{}_fintune_checkpoint.pth'.format(args.subject)
    torch.save(pipe.diffusion_prior.state_dict(), path)

def eval():
    test_data = np.load(args.root_path + '/test_sample_diffprior.npz', allow_pickle=True)
    test_fMRI_hf = test_data['test_fMRI_hf'].astype(np.float32)  # 982,1024

    diffusion_prior = DiffusionPriorUNet(embed_dim=1024, cond_dim=1024, dropout=0.1)

    print(sum(p.numel() for p in diffusion_prior.parameters() if p.requires_grad))
    pipe = Pipe(diffusion_prior, device='cpu')
    path = './checkpoints/diffusion_prior/{}_fintune_checkpoint.pth'.format(args.subject)
    pipe.diffusion_prior.load_state_dict(torch.load(path, map_location='cpu'))
    
    generator = Generator4Embeds(num_inference_steps=4, device='cpu')
    
    wodp_directory = f"generated_images/{args.subject}/wodp_image"
    wdp_directory = f"generated_images/{args.subject}/wdp_image"
    for k in range(0,50):
        fMRI_embeds = torch.tensor(test_fMRI_hf[k:k+1])
        h = pipe.generate(c_embeds=fMRI_embeds,num_inference_steps=50,guidance_scale=5.0)
        for j in range(1):
            wodp_image = generator.generate(fMRI_embeds.to(dtype=torch.float32))
            path = f'{wodp_directory}/{k}_{j}.png'
            os.makedirs(os.path.dirname(path), exist_ok=True)
            wodp_image.save(path)
            
            wdp_image = generator.generate(h.to(dtype=torch.float32))
            path = f'{wdp_directory}/{k}_{j}.png'
            os.makedirs(os.path.dirname(path), exist_ok=True)
            wdp_image.save(path)

def wtext_eval():
    test_data = np.load(args.root_path + '/test_sample_diffprior.npz', allow_pickle=True)
    test_fMRI_hf = test_data['test_fMRI_hf'].astype(np.float32)  # 982,1024
    with open('semantic_level_caption.txt', 'r', encoding='utf-8') as f:
        text_prompt = f.read().split('\n')[:-1]

    diffusion_prior = DiffusionPriorUNet(embed_dim=1024, cond_dim=1024, dropout=0.1)

    print(sum(p.numel() for p in diffusion_prior.parameters() if p.requires_grad))
    pipe = Pipe(diffusion_prior, device='cpu')
    path = './checkpoints/diffusion_prior/{}_fintune_checkpoint.pth'.format(args.subject)
    pipe.diffusion_prior.load_state_dict(torch.load(path, map_location='cpu'))
    
    generator = Generator4Embeds(num_inference_steps=4, device='cpu')
    wtext_directory = f"generated_images/{args.subject}/wtext_image"
    for k in range(0,50):
        fMRI_embeds = torch.tensor(test_fMRI_hf[k:k+1])
        h = pipe.generate(c_embeds=fMRI_embeds,num_inference_steps=50,guidance_scale=5.0)
        for j in range(1):
            wtext_image = generator.generate(h.to(dtype=torch.float32), text_prompt[k])
            path = f'{wtext_directory}/{k}_{j}.png'
            os.makedirs(os.path.dirname(path), exist_ok=True)
            wtext_image.save(path)

if __name__ == '__main__':
    train()
    
    from models.custom_pipeline import *
    eval()
    wtext_eval()

