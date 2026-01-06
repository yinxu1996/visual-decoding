import os
import sys
import numpy as np
import pandas as pd
import random
random.seed(42)
import re
from tqdm import tqdm
import h5py
import json
import scipy.io as spio
import nibabel as nib
from pycocotools.coco import COCO

import argparse
parser = argparse.ArgumentParser(description='Argument Parser')
parser.add_argument("-path", "--path", help="NSD path", default="dataset/NSD/")
parser.add_argument("-sub", "--sub", help="Subject Number", default=1)
args = parser.parse_args()
sub=int(args.sub)
assert sub in [1,2,5,7]
path = args.path
# dorsal + ventral
dorsal_roi = ['V1d', 'V2d', 'V3d', 'OPA', 'EBA']
ventral_roi = ['V1v', 'V2v', 'V3v', 'hV4', 'OFA', 'FFA', 'OWFA', 'VWFA', 'FBA']
roi_name = dorsal_roi + ventral_roi

vocabs_cate = json.load(open('dataset/NSD/processed_data/vocabs_cate.json', "r", encoding="utf-8"))
category_to_index = vocabs_cate['cat2id']
vocabs_smt = json.load(open('dataset/NSD/processed_data/vocabs_smt.json', "r", encoding="utf-8"))
name_to_index = vocabs_smt['smt2id']

def loadmat(filename):
    '''
    this function should be called instead of direct spio.loadmat
    as it cures the problem of not properly recovering python dictionaries
    from mat files. It calls the function check keys to cure all entries
    which are still mat-objects
    '''
    def _check_keys(d):
        '''
        checks if entries in dictionary are mat-objects. If yes
        todict is called to change them to nested dictionaries
        '''
        for key in d:
            if isinstance(d[key], spio.matlab.mio5_params.mat_struct):
                d[key] = _todict(d[key])
        return d

    def _todict(matobj):
        '''
        A recursive function which constructs from matobjects nested dictionaries
        '''
        d = {}
        for strg in matobj._fieldnames:
            elem = matobj.__dict__[strg]
            if isinstance(elem, spio.matlab.mio5_params.mat_struct):
                d[strg] = _todict(elem)
            elif isinstance(elem, np.ndarray):
                d[strg] = _tolist(elem)
            else:
                d[strg] = elem
        return d

    def _tolist(ndarray):
        '''
        A recursive function which constructs lists from cellarrays
        (which are loaded as numpy ndarrays), recursing into the elements
        if they contain matobjects.
        '''
        elem_list = []
        for sub_elem in ndarray:
            if isinstance(sub_elem, spio.matlab.mio5_params.mat_struct):
                elem_list.append(_todict(sub_elem))
            elif isinstance(sub_elem, np.ndarray):
                elem_list.append(_tolist(sub_elem))
            else:
                elem_list.append(sub_elem)
        return elem_list
    data = spio.loadmat(filename, struct_as_record=False, squeeze_me=True)
    return _check_keys(data)

if __name__ == "__main__":

    mask_root = path + 'nsddata/ppdata/'

    stim_order_f = path + 'nsddata/experiments/nsd/nsd_expdesign.mat'
    exp_design = loadmat(stim_order_f)

    nsd_stiminfo_file = path+"nsddata/experiments/nsd/nsd_stim_info_merged.pkl"
    stiminfo = pd.read_pickle(nsd_stiminfo_file)  

    subject_idx  = exp_design['subjectim'] 
    trial_order  = exp_design['masterordering']  
    cocoId_arr = np.zeros(shape=subject_idx.shape, dtype=int) 
    for j in range(len(subject_idx)):
        cocoId = np.array(stiminfo['cocoId'])[stiminfo['subject%d'%(j+1)].astype(bool)]
        nsdId = np.array(stiminfo['nsdId'])[stiminfo['subject%d'%(j+1)].astype(bool)]
        imageId = subject_idx[j]-1
        for i,k in enumerate(imageId):
            cocoId_arr[j,i] = (cocoId[nsdId==k])[0]
    
    # Selecting ids for training and test data
    sig_train = {}
    sig_train_cocoId = {}
    sig_test = {}
    sig_test_cocoId = {}
    num_trials = 37*750 
    for idx in range(num_trials):
        ''' nsdId as in design csv files'''
        nsdId = subject_idx[sub-1, trial_order[idx] - 1] - 1
        info = stiminfo[stiminfo['nsdId']==nsdId]
        if info['shared1000'].iloc[0]:
            if nsdId not in sig_test:
                sig_test[nsdId] = []
            sig_test[nsdId].append(idx)
            if nsdId not in sig_test_cocoId:
                sig_test_cocoId[nsdId] = []
            sig_test_cocoId[nsdId].append(info['cocoId'].iloc[0])
        else:
            if nsdId not in sig_train:
                sig_train[nsdId] = []
            sig_train[nsdId].append(idx)
            if nsdId not in sig_train_cocoId:
                sig_train_cocoId[nsdId] = []
            sig_train_cocoId[nsdId].append(info['cocoId'].iloc[0])

    train_im_idx = list(sig_train.keys())
    train_trial_idx = [trial for trials in sig_train.values() for trial in trials]  # trail id: 24980
    test_im_idx = list(sig_test.keys())
    test_trial_idx = [trial for trials in sig_test.values() for trial in trials]

    '''
    save fmri
    '''
    roi_dir = path + 'nsddata/ppdata/subj{:02d}/func1pt8mm/roi/'.format(sub)
    betas_dir = path + 'nsddata_betas/ppdata/subj{:02d}/func1pt8mm/betas_fithrf_GLMdenoise_RR/'.format(sub)
    
    # D:V1d, V2d, V3d; V:V1v, V2v, V3v, hV4
    voxel_prf_visualrois = nib.load(mask_root + "subj%02d/func1pt8mm/roi/prf-visualrois.nii.gz"%(sub)).get_fdata()
    V1v_mask = (voxel_prf_visualrois==1) 
    V1d_mask = (voxel_prf_visualrois==2)
    V2v_mask = (voxel_prf_visualrois==3) 
    V2d_mask = (voxel_prf_visualrois==4)
    V3v_mask = (voxel_prf_visualrois==5) 
    V3d_mask = (voxel_prf_visualrois==6)
    hV4_mask = (voxel_prf_visualrois==7)
    # V:OFA, FFA
    voxel_floc_faces = nib.load(mask_root + "subj%02d/func1pt8mm/roi/floc-faces.nii.gz"%(sub)).get_fdata()
    OFA_mask = (voxel_floc_faces==1)
    FFA_mask = (voxel_floc_faces==2) | (voxel_floc_faces==3)
    # V:OWFA, VWFA
    voxel_floc_words = nib.load(mask_root + "subj%02d/func1pt8mm/roi/floc-words.nii.gz"%(sub)).get_fdata()
    OWFA_mask = (voxel_floc_words==1)
    VWFA_mask = (voxel_floc_words==2) | (voxel_floc_faces==3)
    # D:OPA
    voxel_floc_places = nib.load(mask_root + "subj%02d/func1pt8mm/roi/floc-places.nii.gz"%(sub)).get_fdata()
    OPA_mask = (voxel_floc_places==1)
    # D:EBA; V:FBA
    voxel_floc_bodies = nib.load(mask_root + "subj%02d/func1pt8mm/roi/floc-bodies.nii.gz"%(sub)).get_fdata()
    EBA_mask = (voxel_floc_bodies==1)
    FBA_mask = (voxel_floc_bodies==2) | (voxel_floc_bodies==3)
    
    roi_mask = np.stack((V1d_mask, V2d_mask, V3d_mask, OPA_mask, EBA_mask,
                         V1v_mask, V2v_mask, V3v_mask, hV4_mask, OFA_mask, FFA_mask, OWFA_mask, VWFA_mask, FBA_mask)
                         , axis=0)
    
    roi_vn = {}
    roi_mask_index = {}
    roi_mask_fmri = {}
    for i, name in enumerate(roi_name):
        mask = roi_mask[i]
        print ("%s \t: %d" % (name, np.sum(mask)))
        roi_vn[name] = np.sum(mask)
        roi_mask_index[name] = mask
        roi_mask_fmri[name] = np.zeros((num_trials, np.sum(mask))).astype(np.float32)
    np.savez(path + 'processed_data/subj{:02d}/roi_vn.npz'.format(sub), **roi_vn)
    np.save(path + 'processed_data/subj{:02d}/roi_mask_index.npy'.format(sub), roi_mask_index)    
    
    for i in tqdm(range(37), desc="Processing"):  # 37 sessions
        beta_filename = "betas_session{0:02d}.nii.gz".format(i+1)
        beta_f = nib.load(betas_dir+beta_filename).get_fdata().astype(np.float32)  # (81, 104, 83, 750)  each voxel's 750 trials
        for name in roi_name:
            roi_mask = roi_mask_index[name]
            roi_mask_fmri[name][i*750:(i+1)*750] = beta_f[roi_mask].transpose()
        del beta_f
    print('roi mask fmri data are loaded.')

    num_train, num_test = len(train_trial_idx), len(test_im_idx)
    for name in roi_name:
        vox_dim = roi_mask_fmri[name].shape[1]
        fmri = roi_mask_fmri[name]
        fmri_array = np.zeros((num_train,vox_dim))
        for i, idx in enumerate(train_trial_idx):
            fmri_array[i] = fmri[idx]   
        np.save(path + 'processed_data/subj{:02d}/train_roi_mask_{}_raw.npy'.format(sub,name),fmri_array)

        fmri_array = np.zeros((num_test,vox_dim))
        for i, idx in enumerate(test_im_idx):
            fmri_array[i] = fmri[sorted(sig_test[idx])].mean(0)
        np.save(path + 'processed_data/subj{:02d}/test_roi_mask_{}_raw.npy'.format(sub,name),fmri_array)
    print("fMRI data is saved.")
    
    '''
    save images
    '''
    f_stim = h5py.File(path + 'nsddata_stimuli/stimuli/nsd/nsd_stimuli.hdf5', 'r')
    stim = f_stim['imgBrick'][:]
    print('Stimuli are loaded.')

    num_train, num_test = len(train_im_idx), len(test_im_idx)
    im_dim, im_c = 425, 3
    stim_array = np.zeros((num_train, im_dim, im_dim, im_c))
    for i, idx in enumerate(train_im_idx):
        stim_array[i] = stim[idx]
    np.save(path + 'processed_data/subj{:02d}/train_stim.npy'.format(sub), stim_array)

    fmri_counts = [len(fmri_indices) for fmri_indices in sig_train.values()]
    np.save(path + 'processed_data/subj{:02d}/train_fmri_to_image_counts.npy'.format(sub), fmri_counts)
    
    stim_array = np.zeros((num_test, im_dim, im_dim, im_c))
    for i,idx in enumerate(test_im_idx):
        stim_array[i] = stim[idx]
    np.save(path + 'processed_data/subj{:02d}/test_stim.npy'.format(sub), stim_array)
    print('Stimuli data is saved.')
    
    '''
    save supercategories and name
    '''
    cat = np.load(path + '/processed_data/image_info.npy', allow_pickle=True).item()

    train_ids = [x[0] for x in sig_train_cocoId.values()]
    temp = [cat[i] for i in train_ids]
    coco_cat = []
    for bc in temp:
        area = bc[0]['area']
        cc = bc[0]['supercategory']
        for bb in bc:
            if area < bb['area']:
                area = bb['area']
                cc = bb['supercategory']
        coco_cat.append(cc)
    supercategories = np.array([category_to_index[i] for i in coco_cat])
    np.save(path + 'processed_data/subj{:02d}/train_supercategories.npy'.format(sub), supercategories)

    coco_smt = [list(set([x['name'] for x in cat[i]])) for i in train_ids]
    names = [[name_to_index[j] for j in i] for i in coco_smt]
    name_onehot_list = []
    for i in names:
        name_onehot = np.zeros(80,)
        name_onehot[i] = 1
        name_onehot_list.append(name_onehot)
    name_onehot_list = np.array(name_onehot_list)
    np.save(path + 'processed_data/subj{:02d}/train_names.npy'.format(sub), name_onehot_list)
    
    # ================================================================
    test_ids = [x[0] for x in sig_test_cocoId.values()]
    temp = [cat[i] for i in test_ids]
    coco_cat = []
    for bc in temp:
        area = bc[0]['area']
        cc = bc[0]['supercategory']
        for bb in bc:
            if area < bb['area']:
                area = bb['area']
                cc = bb['supercategory']
        coco_cat.append(cc)
    supercategories = np.array([category_to_index[i] for i in coco_cat])
    np.save(path + 'processed_data/subj{:02d}/test_supercategories.npy'.format(sub), supercategories)

    coco_smt = [list(set([x['name'] for x in cat[i]])) for i in test_ids]
    names = [[name_to_index[j] for j in i] for i in coco_smt]
    name_onehot_list = []
    for i in names:
        name_onehot = np.zeros(80,)
        name_onehot[i] = 1
        name_onehot_list.append(name_onehot)
    name_onehot_list = np.array(name_onehot_list)
    np.save(path + 'processed_data/subj{:02d}/test_names.npy'.format(sub), name_onehot_list)

