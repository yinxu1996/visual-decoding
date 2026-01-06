import argparse
import torch
import open_clip
import numpy as np
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from PIL import Image
from tqdm import tqdm
import torchvision.models as models

parser = argparse.ArgumentParser(description='CLIP Image Embedding')
parser.add_argument('--device', type=str, default='cuda:0', help='Device to use (e.g., cuda:0, cpu)')
parser.add_argument('--sub', type=int, default='1', help='Subject number')
parser.add_argument('--path', type=str, default='dataset/NSD/', help='Base path for data')
args = parser.parse_args()

class CLIPImageEmbedding:
    def __init__(self, device='cuda'):
        self.device = device
        self.model, self.preprocess_train, self.feature_extractor = open_clip.create_model_and_transforms(
            'ViT-H/14', pretrained='laion2b_s32b_b79k', precision='fp32', device = device)

        self.transform = Compose([
            Resize((224, 224)),
            ToTensor(),
            Normalize((0.48145466, 0.4578275, 0.40821073), 
                     (0.26862954, 0.26130258, 0.27577711))
        ])
    
    def extract_high_level_features(self, image_path, sub):
        images = np.load(image_path.format(sub))
        
        features = []
        for img in tqdm(images, desc="Processing images"):
            img_pil = Image.fromarray((img).astype(np.uint8))
            img_tensor = self.transform(img_pil).unsqueeze(0).to(self.device)
            with torch.no_grad():
                image_features = self.model.encode_image(img_tensor)
            features.append(image_features.cpu().numpy())
        
        return np.concatenate(features, axis=0)

class VGGImageEmbedding:
    def __init__(self, device='cuda'):
        self.device = device
        self.vgg = models.vgg16(pretrained=True).features.to(device)
        self.vgg.eval()
        
        self.transform = Compose([
            Resize((224, 224)),
            ToTensor(),
            Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
    
    def extract_low_level_features(self, image_path, sub):
        images = np.load(image_path.format(sub))
        features = []
        for img in tqdm(images, desc="Processing images"):
            img_pil = Image.fromarray((img).astype(np.uint8))
            img_tensor = self.transform(img_pil).unsqueeze(0).to(self.device)
            with torch.no_grad():
                low_level = self.vgg(img_tensor)
                low_level = low_level.mean(dim=[2, 3])
                features.append(low_level.cpu().numpy())
        
        return np.concatenate(features, axis=0)

if __name__ == "__main__":
    '''Train'''
    clip_embedding = CLIPImageEmbedding(device=args.device)
    train_high_level = clip_embedding.extract_high_level_features(
        args.path + 'processed_data/subj{:02d}/train_stim.npy',
        sub=args.sub
    )
    
    vgg_embedding = VGGImageEmbedding(device=args.device)
    train_low_level = vgg_embedding.extract_low_level_features(
        args.path + 'processed_data/subj{:02d}/train_stim.npy',
        sub=args.sub
    )
    
    np.save(args.path + 'processed_data/subj{:02d}/train_high-level_features.npy'.format(args.sub), train_high_level)
    np.save(args.path + 'processed_data/subj{:02d}/train_low-level_features.npy'.format(args.sub), train_low_level)

    '''test'''
    clip_embedding = CLIPImageEmbedding(device=args.device)
    test_high_level = clip_embedding.extract_high_level_features(
        args.path + 'processed_data/subj{:02d}/test_stim.npy',
        sub=args.sub
    )
    
    vgg_embedding = VGGImageEmbedding(device=args.device)
    test_low_level = vgg_embedding.extract_low_level_features(
        args.path + 'processed_data/subj{:02d}/test_stim.npy',
        sub=args.sub
    )
    
    np.save(args.path + 'processed_data/subj{:02d}/test_high-level_features.npy'.format(args.sub), test_high_level)
    np.save(args.path + 'processed_data/subj{:02d}/test_low-level_features.npy'.format(args.sub), test_low_level)