import os
import numpy as np
import re
import random
import json
import torch
from torch.utils.data import Dataset
from data_provider.uea import normalize_batch_ts
import warnings
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

warnings.filterwarnings("ignore")

class NSDLoader(Dataset):
    def __init__(self, root_path, supercategories, roi_names, flag):
        self.root_path = root_path
        self.supercategories = supercategories
        self.roi_names = roi_names
        self.flag = flag
        self.fmri, self.hf, self.lf, self.cate, \
        self.name, self.sample_index = self.load_NSD(self.root_path, 
        self.supercategories, self.roi_names, self.flag)
        
    def load_NSD(self, data_path, supercategories, roi_names, flag):  
        filenames = []
        for filename in os.listdir(data_path):
            filenames.append(filename)
        filenames.sort()
        filtered_files = [filename for filename in filenames if flag in filename]
        fmri_list = []
        for roi in roi_names:
            for j in range(len(filtered_files)):
                path = data_path + filtered_files[j]
                if roi in filtered_files[j]: 
                    if 'vs' in filtered_files[j]:
                        data = np.load(path)
                        fmri_list.append(data)
        fmri = np.array(fmri_list).transpose(1,0,2)

    
        if flag == 'train':
            fmri_to_image_counts = np.load(data_path + 'train_fmri_to_image_counts.npy')
            for j in range(len(filtered_files)):
                path = data_path + filtered_files[j]
                if 'high-level_features' in filtered_files[j]:
                    data = np.load(path)
                    high_features = np.repeat(data, fmri_to_image_counts, axis=0)
                    high_features = scaler.fit_transform(high_features)
                elif 'low-level_features' in filtered_files[j]:
                    data = np.load(path)
                    low_features = np.repeat(data, fmri_to_image_counts, axis=0)
                    low_features = scaler.fit_transform(low_features)
                elif 'supercategories' in filtered_files[j]:
                    data = np.load(path)
                    cate = np.repeat(data, fmri_to_image_counts, axis=0)
                    cate = np.expand_dims(cate, axis=1)
                elif 'name' in filtered_files[j]:
                    data = np.load(path)
                    name = np.repeat(data, fmri_to_image_counts, axis=0)
        elif flag == 'test':
            for j in range(len(filtered_files)):
                path = data_path + filtered_files[j]
                if 'high-level_features' in filtered_files[j]:
                    high_features = np.load(path)
                    high_features = scaler.fit_transform(high_features)
                elif 'low-level_features' in filtered_files[j]:
                    low_features = np.load(path)
                    low_features = scaler.fit_transform(low_features)  
                elif 'supercategories' in filtered_files[j]:
                    cate = np.load(path)
                    cate = np.expand_dims(cate, axis=1)
                elif 'name' in filtered_files[j]:
                    name = np.load(path)

        sample_index = np.arange(len(name))[:,np.newaxis]
        fmri, high_features, low_features, cate, name, sample_index \
            = shuffle(fmri, high_features, low_features, cate, name, sample_index, random_state=42)
        return fmri, high_features, low_features, cate, name, sample_index
    
    def __getitem__(self, index):
        return torch.from_numpy(self.fmri[index]), torch.from_numpy(self.hf[index]), \
                torch.from_numpy(self.lf[index]), torch.from_numpy(self.cate[index]), \
                torch.from_numpy(self.name[index]), torch.from_numpy(self.sample_index[index])

    def __len__(self):
        return len(self.cate)
