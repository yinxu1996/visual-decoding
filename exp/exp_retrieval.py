from tracemalloc import start
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import precision_recall_curve, auc
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import os
import time
import warnings
import numpy as np
import random
import matplotlib.pyplot as plt
import json
from tqdm import tqdm
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from sklearn.metrics.pairwise import cosine_similarity
from data_provider.data_loader import (
    NSDLoader,
)

data_dict = {
    "NSD": NSDLoader,
}

warnings.filterwarnings("ignore")

class Exp_Retrieval(Exp_Basic):
    def __init__(self, args):
        super().__init__(args)
        self.smoothing_function = SmoothingFunction().method1

    def _build_model(self):
        model = (
            self.model_dict[self.args.model].Transformer_Retrieval(self.args).float()
        )
        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform(p)

        return model
    
    def _get_data(self, flag):
        data_set, data_loader = self.data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate, betas=(0.9, 0.98), eps=1e-9)
        return model_optim
    
    def data_provider(self, args, flag):
        Data = data_dict[args.data]

        data_set = Data(
            root_path=args.root_path,
            supercategories=args.supercategories,
            roi_names = args.roi_name,
            flag=flag,
        )

        data_loader = DataLoader(
            data_set,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            drop_last=False
        )
        return data_set, data_loader

    def vali(self, val_data, val_loader):
        self.model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for i, (fmri, hf, lf, cate, name, _) in enumerate(val_loader):
                batch_fmri = fmri.float().to(self.device)
                batch_hf = hf.float().to(self.device)
                batch_lf = lf.float().to(self.device)
                batch_cate = cate.squeeze().to(self.device)
                batch_name = name.float().to(self.device)

                contrastive_loss, mse_loss, ortho_loss,\
                    hf_pred, lf_pred = self.model(batch_fmri, batch_hf, batch_lf)
                loss = contrastive_loss + mse_loss + ortho_loss

                val_loss += loss

            self.model.train()
            return val_loss / len(val_loader)
    
    def train(self):
        print(">>>>>>>start training: {}, {}, seed{} >>>>>>>>>>>>>>>>>>>>>>>>>>".format(self.args.task_name, self.args.subject, self.args.seed))
        
        train_data, train_loader = self._get_data(flag="train")
        test_data, test_loader = self._get_data(flag="test")

        path = ("./checkpoints/" + self.args.task_name + "/" + self.args.model + "/")
        if not os.path.exists(path):
            os.makedirs(path)

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(
            patience=self.args.patience, verbose=True, delta=1e-5
        )

        model_optim = self._select_optimizer()

        for epoch in range(self.args.train_epochs):
            train_loss = 0.0

            self.model.train()

            start = time.time()
            for i, (fmri, hf, lf, cate, name, _) in enumerate(train_loader):
                model_optim.zero_grad()

                batch_fmri = fmri.float().to(self.device)
                batch_hf = hf.float().to(self.device)
                batch_lf = lf.float().to(self.device)
                batch_cate = cate.squeeze().to(self.device)
                batch_name = name.float().to(self.device)
                
                contrastive_loss, mse_loss, ortho_loss, hf_pred, lf_pred = self.model(batch_fmri, batch_hf, batch_lf)
                loss = contrastive_loss + mse_loss + ortho_loss

                train_loss += loss

                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10)
                model_optim.step()
            
            end = time.time()
            train_loss = train_loss / len(train_loader)
            val_loss = self.vali(test_data, test_loader)

            if epoch % 1 == 0:
                print(
                    f"Epoch: {epoch + 1}, Steps: {train_steps},\n"
                    f"Train Loss: {train_loss:.5f} | Validation Loss: {val_loss:.5f} | Train time: {end - start:.2f}s"
                )

            early_stopping(
                val_loss,
                self.model,
                path,
                self.args.subject,
                self.args.seed
            )
            if early_stopping.early_stop:
                print("Early stopping")
                break

    def _fmri_embedding(self, query_fmri):
        query_fmri = torch.tensor(query_fmri).float().to(self.device)
        batch_size = query_fmri.size(0)

        src = self.model.multiscale_fmri_embed(query_fmri)
        ls_brain_token = self.model.ls_brain_token.expand(batch_size, -1, -1)
        ms_brain_token = self.model.ms_brain_token.expand(batch_size, -1, -1)
        ss_brain_token = self.model.ss_brain_token.expand(batch_size, -1, -1)
        high_token = self.model.high_feature_token.expand(batch_size, -1, -1)
        low_token = self.model.low_feature_token.expand(batch_size, -1, -1)
        src = torch.cat((ls_brain_token, ms_brain_token, ss_brain_token, src, high_token, low_token), dim=1)
        src *= 2 * src.shape[-1] ** 0.5
        src += self.model.a_fmri * self.model.positional_encoding(src, src.shape[1]).to(self.device)
        
        memory = src
        for layer in self.model.encoder_layers:
            multi_scale_self_attn_output, _ = layer['multiscale_self_attn'](memory.permute(1, 0, 2))
            multi_scale_self_attn_output = multi_scale_self_attn_output.permute(1, 0, 2)
            multi_scale_self_attn_output = layer['layer1_norm1'](memory + layer['layer1_dropout'](multi_scale_self_attn_output))
            multi_scale_self_attn_output = layer['layer1_norm2'](multi_scale_self_attn_output + layer['layer1_dropout'](layer['layer1_ffn'](multi_scale_self_attn_output)))
            
            brain_token, high_token, low_token = multi_scale_self_attn_output[:,:-2,:], multi_scale_self_attn_output[:,-2,:], multi_scale_self_attn_output[:,-1,:]
            
            high_token, low_token, g_h, g_l = self.model.gated(high_token, low_token)  # gated token modulator

            cross_attn1_output, _ = layer['cross_attn1'](
                query=torch.cat([high_token.unsqueeze(1), low_token.unsqueeze(1)], dim=1).permute(1, 0, 2),  # (2, batch_size, embed_dim)
                key=brain_token.permute(1, 0, 2),
                value=brain_token.permute(1, 0, 2)
            )
            cross_attn1_output = cross_attn1_output.permute(1, 0, 2)
            cross_attn1_output = layer['layer2_norm1'](torch.cat([high_token.unsqueeze(1), low_token.unsqueeze(1)], dim=1) + layer['layer2_dropout'](cross_attn1_output))
            cross_attn1_output = layer['layer2_norm2'](cross_attn1_output + layer['layer2_dropout'](layer['layer2_ffn'](cross_attn1_output)))
            memory  = torch.cat([brain_token, cross_attn1_output], dim=1)

        query_hf = self.model.high_fc(high_token)
        query_lf = self.model.low_fc(low_token)
        
        return query_hf, query_lf

    def voc_ap(self, rec, prec, true_num):
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
        i = np.where(mrec[1:] != mrec[:-1])[0]
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
        return ap
    
    def mAP(self, predict, target):
        class_num = target.shape[1]
        seg = np.concatenate((predict, target), axis=1)
        gt_label = seg[:, class_num:].astype(np.int32)
        num_target = np.sum(gt_label, axis=1, keepdims=True)
        threshold = 1 / (num_target + 1e-6)
        predict_result = seg[:, 0:class_num] > threshold
        sample_num = len(gt_label)
        tp = np.zeros(sample_num)
        fp = np.zeros(sample_num)
        aps = []
        recall = []
        precise = []
        for class_id in range(class_num):
            confidence = seg[:, class_id]
            sorted_ind = np.argsort(-confidence)
            sorted_scores = np.sort(-confidence)
            sorted_label = [gt_label[x][class_id] for x in sorted_ind]
            for i in range(sample_num):
                tp[i] = (sorted_label[i] > 0)
                fp[i] = (sorted_label[i] <= 0)
            true_num = sum(tp)
            fp = np.cumsum(fp)
            tp = np.cumsum(tp)
            if true_num == 0:
                rec = tp / 1000000
            else:
                rec = tp / float(true_num)
            prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
            recall += [rec]
            precise += [prec]
            ap = self.voc_ap(rec, prec, true_num)
            aps += [ap]
        np.set_printoptions(precision=3, suppress=True)

        mAPvalue = np.mean(aps)
        return mAPvalue, aps

    def results(self):
        print(">>>>>>>retrieval task results: {}, {}, seed{} >>>>>>>>>>>>>>>>>>>>>>>>>>".format(self.args.task_name, self.args.subject, self.args.seed))
        
        best_model_path = "checkpoints/{}/{}/{}_seed{}_checkpoint.pth".format(self.args.task_name,self.args.model,self.args.subject,self.args.seed)
        self.model.load_state_dict(torch.load(best_model_path))
        self.model.eval()

        test_data, test_loader = self._get_data(flag="test")
        fmri = torch.tensor(test_data.fmri).float().to(self.device)
        hf = torch.tensor(test_data.hf).float().to(self.device)
        lf = torch.tensor(test_data.lf).float().to(self.device)
        cate = test_data.cate
        name = test_data.name
        sample_index = test_data.sample_index

        for top_k in [1,5,10]:
            num_samples = len(fmri)
            cate_correct_count = 0
            total_precision = 0
            retrieval_correct_count = 0
            success_retrieval_indices_list = []
            pred_labels_list = []
            query_hf_all_samples = []
            query_lf_all_samples = []
            pred_cate = []
            for i in range(num_samples):
                query_fmri = fmri[i:i+1]
                query_hf, query_lf = self._fmri_embedding(query_fmri)
                hf_similarity_scores = F.cosine_similarity(query_hf, hf, dim=-1).squeeze()
                lf_similarity_scores = F.cosine_similarity(query_lf, lf, dim=-1).squeeze()
                if top_k == 1:
                    query_hf_all_samples.append(query_hf.squeeze().cpu().detach().numpy())
                    query_lf_all_samples.append(query_lf.squeeze().cpu().detach().numpy())
                
                similarity_scores = (hf_similarity_scores + lf_similarity_scores) / 2.0
                # similarity_scores = hf_similarity_scores
                _, retrieved_indices = torch.topk(similarity_scores, k=top_k)
                retrieved_indices = retrieved_indices.cpu().detach().numpy()
                
                pred_cate.append(cate[retrieved_indices[0]][0])
                if cate[i] in cate[retrieved_indices]:
                    cate_correct_count += 1
                
                pred_labels_list.append(np.sum(name[retrieved_indices],axis=0))
                
                if i in retrieved_indices:
                    retrieval_correct_count += 1
                    success_retrieval_indices_list.append(np.insert(retrieved_indices, 0, i)) 
           
            classification_accuracy = cate_correct_count / num_samples
            
            pred_labels_list = np.array(pred_labels_list).squeeze()
            mAPvalue, aps = self.mAP(pred_labels_list, name)

            retrieval_rate = retrieval_correct_count / num_samples
            print(
                f"Classification Accuracy (top_k={top_k}): {classification_accuracy:.5f} \n"
                f"mPA (top_k={top_k}): {mAPvalue:.5f} \n"
                f"Retrieval Rate (top_k={top_k}): {retrieval_rate:.5f}"
            )

    def save_feature(self):
        print(">>>>>>>save feature: {}, {}, seed{} >>>>>>>>>>>>>>>>>>>>>>>>>>".format(self.args.task_name, self.args.subject, self.args.seed))

        best_model_path = "checkpoints/{}/{}/{}_seed{}_checkpoint.pth".format(self.args.task_name,self.args.model,self.args.subject,self.args.seed)
        self.model.load_state_dict(torch.load(best_model_path))
        self.model.eval()
        train_data, train_loader = self._get_data(flag="train")
        train_fMRI_hf_list = []
        train_fMRI_lf_list = []
        train_image_hf_list = []
        train_image_lf_list = []
        train_sample_index_list = []
        for i, (fmri, hf, lf, cate, name, sample_index) in enumerate(train_loader):
                batch_fmri = fmri.float().to(self.device)
                batch_fMRI_hf, batch_fMRI_lf = self._fmri_embedding(batch_fmri)
                train_fMRI_hf_list.append(batch_fMRI_hf.cpu().detach().numpy())
                train_fMRI_lf_list.append(batch_fMRI_lf.cpu().detach().numpy())
                train_image_hf_list.append(hf)
                train_image_lf_list.append(lf)
                train_sample_index_list.append(sample_index)
        train_fMRI_hf = np.concatenate(train_fMRI_hf_list, axis=0)
        train_fMRI_lf = np.concatenate(train_fMRI_lf_list, axis=0)
        train_image_hf = np.concatenate(train_image_hf_list, axis=0)
        train_image_lf = np.concatenate(train_image_lf_list, axis=0)
        train_sample_index = np.concatenate(train_sample_index_list, axis=0) 

        np.savez(self.args.root_path + '/train_sample_diffprior.npz', 
            train_fMRI_hf=train_fMRI_hf, train_fMRI_lf=train_fMRI_lf, 
            train_image_hf=train_image_hf, train_image_lf=train_image_lf,
            train_sample_index=train_sample_index)
        print('train sample done')
        
        test_data, test_loader = self._get_data(flag="test")
        test_fmri = torch.tensor(test_data.fmri).float().to(self.device)
        test_image_hf = test_data.hf
        test_image_lf = test_data.lf
        test_sample_index = test_data.sample_index

        test_fMRI_hf, test_fMRI_lf = self._fmri_embedding(test_fmri)
        test_fMRI_hf = test_fMRI_hf.cpu().detach().numpy()
        test_fMRI_lf = test_fMRI_lf.cpu().detach().numpy()
        np.savez(self.args.root_path + '/test_sample_diffprior.npz', 
            test_fMRI_hf=test_fMRI_hf, test_fMRI_lf=test_fMRI_lf, 
            test_image_hf=test_image_hf, test_image_lf=test_image_lf,
            test_sample_index=test_sample_index)
        print('test sample done')
    


