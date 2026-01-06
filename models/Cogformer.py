import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class GatedTokenModulator(nn.Module):
    def __init__(self, dim):
        super(GatedTokenModulator, self).__init__()
        self.gate_mlp = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
            nn.Sigmoid()
        )

    def forward(self, z_h, z_l):
        g_h = self.gate_mlp(z_h)
        g_l = self.gate_mlp(z_l)
        
        z_h_mod = z_h + g_h * z_l
        z_l_mod = z_l + g_l * z_h
        return z_h_mod, z_l_mod, g_h, g_l

class FMRIEmbedding(nn.Module):
    def __init__(self, brain_roi, enc_in, e_model, dropout):
        super(FMRIEmbedding, self).__init__()
        self.dorsal_linear = nn.Linear(enc_in*5, e_model)
        self.ventral_linear = nn.Linear(enc_in*9, e_model)
        self.visualrois_linear = nn.Linear(enc_in*7, e_model)
        self.faces_linear = nn.Linear(enc_in*2, e_model)
        self.words_linear = nn.Linear(enc_in*2, e_model)
        self.places_linear = nn.Linear(enc_in, e_model)
        self.bodies_linear = nn.Linear(enc_in*2, e_model)
        self.small_linear = nn.ModuleList([nn.Linear(enc_in, e_model) for _ in range(brain_roi)])
        self.layernorm = nn.LayerNorm(e_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, dorsal_src, ventral_src, 
                visualrois_src, faces_src, words_src, places_src, bodies_src,
                src):
        dorsal_x = torch.unsqueeze(self.dorsal_linear(dorsal_src), dim=1)
        ventral_x = torch.unsqueeze(self.ventral_linear(ventral_src), dim=1)
        
        visualrois_x = torch.unsqueeze(self.visualrois_linear(visualrois_src), dim=1)
        faces_x = torch.unsqueeze(self.faces_linear(faces_src), dim=1)
        words_x = torch.unsqueeze(self.words_linear(words_src), dim=1)
        places_x = torch.unsqueeze(self.places_linear(places_src), dim=1)
        bodies_x = torch.unsqueeze(self.bodies_linear(bodies_src), dim=1)
        rois_x = []
        for i, linear_layer in enumerate(self.small_linear):
            roi_x = linear_layer(src[:,i,:])
            rois_x.append(roi_x)
        rois_x = torch.stack(rois_x, dim=1)

        x = torch.concatenate((self.dropout(dorsal_x), self.dropout(ventral_x),
                                self.dropout(visualrois_x), self.dropout(faces_x), self.dropout(words_x), self.dropout(places_x), self.dropout(bodies_x),
                                self.dropout(rois_x)), dim=1)
        
        x = self.layernorm(x)
        return x

class Cate(nn.Module):
    def __init__(self, hidden, supercategories):
        super(Cate, self).__init__()
        self.cf1 = nn.Linear(hidden, int(hidden/2))
        self.cf2 = nn.Linear(int(hidden/2), supercategories)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()

    def forward(self, x):
        x = self.relu(self.cf1(x))
        x_feature = self.cf2(x)
        x = self.softmax(x_feature)
        return x_feature, x
    
class Name(nn.Module):
    def __init__(self, hidden, names):
        super(Name, self).__init__()
        self.cf1 = nn.Linear(hidden, int(hidden/2))
        self.cf2 = nn.Linear(int(hidden/2), names)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        x = self.relu(self.cf1(x))
        x = self.sigmoid(self.cf2(x))
        return x

class MultiScaleAttention(nn.Module):
    def __init__(self, hidden_size, num_heads, dropout_prob=0.1):
        super(MultiScaleAttention, self).__init__()
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.attn_layers = nn.ModuleList([
            nn.MultiheadAttention(embed_dim=hidden_size, num_heads=num_heads, dropout=dropout_prob)
            for _ in range(5)
        ])
        self.dropout = nn.Dropout(dropout_prob)

        self.scale_range = [
            [0,3,4],
            [1,5,6,7,8,9],
            [2,10,11,12,13,14,15,16,17,18,19,20,21,22,23]
        ]
        self.inter_scale_range = [0, 1, 2] 

    def forward(self, x):
        attn_weights_list = {}
        for i, scale_range in enumerate(self.scale_range):
            scale_x = x[scale_range, :, :] 
            attn_output, attn_weights = self.attn_layers[i](scale_x, scale_x, scale_x)
            x[scale_range,:,:] = attn_output
            attn_weights_list[f"intra_scale_{i}"] = attn_weights[:,1:,1:]
        
        inter_scale_x = x[self.inter_scale_range]
        inter_scale_output, attn_weights = self.attn_layers[3](inter_scale_x, inter_scale_x, inter_scale_x)
        x[self.inter_scale_range,:,:] = inter_scale_output
        attn_weights_list["inter_scale"] = attn_weights

        x, attn_weights = self.attn_layers[4](x,x,x)
        attn_weights_list["all_scale"] = attn_weights
        
        x = self.dropout(x)
        return x, attn_weights_list

class Transformer_Retrieval(nn.Module):
    def __init__(self, configs):
        super(Transformer_Retrieval, self).__init__()

        self.device = configs.gpu

        self.ss_brain_token = nn.Parameter(torch.zeros(1, 1, configs.e_model))
        self.ms_brain_token = nn.Parameter(torch.zeros(1, 1, configs.e_model))
        self.ls_brain_token = nn.Parameter(torch.zeros(1, 1, configs.e_model))
        
        self.high_feature_token = nn.Parameter(torch.zeros(1, 1, configs.e_model))
        self.low_feature_token = nn.Parameter(torch.zeros(1, 1, configs.e_model))
        
        self.a_fmri = nn.Parameter(torch.zeros(1))

        self.gated = GatedTokenModulator(configs.e_model)

        self.fmri_embed = FMRIEmbedding(
            configs.brain_roi,
            configs.enc_in,
            configs.e_model,
            configs.dropout
        )
        
        self.encoder_layers = nn.ModuleList()
        for _ in range(configs.e_layers):
            multiscale_self_attn = MultiScaleAttention(
                hidden_size=configs.e_model,
                num_heads=configs.n_heads,
                dropout_prob=configs.dropout
            )
            layer1_norm1 = nn.LayerNorm(configs.e_model)
            layer1_ffn = nn.Sequential(
                nn.Linear(configs.e_model, configs.d_ff),
                nn.ReLU(),
                nn.Linear(configs.d_ff, configs.e_model)
            )
            layer1_norm2 = nn.LayerNorm(configs.e_model)
            layer1_dropout = nn.Dropout(configs.dropout)

            cross_attn1 = nn.MultiheadAttention(
                embed_dim=configs.e_model,
                num_heads=configs.n_heads,
                dropout=configs.dropout
            )
            layer2_norm1 = nn.LayerNorm(configs.e_model)
            layer2_ffn = nn.Sequential(
                nn.Linear(configs.e_model, configs.d_ff),
                nn.ReLU(),
                nn.Linear(configs.d_ff, configs.e_model)
            )
            layer2_norm2 = nn.LayerNorm(configs.e_model)
            layer2_dropout = nn.Dropout(configs.dropout)

            self.encoder_layers.append(nn.ModuleDict({
                'multiscale_self_attn': multiscale_self_attn,
                'layer1_norm1': layer1_norm1,
                'layer1_ffn': layer1_ffn,
                'layer1_norm2': layer1_norm2,
                'layer1_dropout': layer1_dropout,

                'cross_attn1': cross_attn1,
                'layer2_norm1': layer2_norm1,
                'layer2_ffn': layer2_ffn,
                'layer2_norm2': layer2_norm2,
                'layer2_dropout': layer2_dropout
            }))

        self.high_fc = nn.Linear(configs.e_model, configs.hf_dim)
        self.low_fc = nn.Linear(configs.e_model, configs.lf_dim)

        # classification
        self.cate = Cate(configs.e_model*2, configs.supercategories)
        self.name = Name(configs.e_model*2, configs.labels)

    def multiscale_fmri_embed(self, src):
        dorsal_src = src[:,:5,:].reshape(src.shape[0], -1)
        ventral_src = src[:,5:,:].reshape(src.shape[0], -1)
        visualrois_src = src[:,[5,0,6,1,7,2,8],:].reshape(src.shape[0], -1)
        faces_src = src[:,[9,10],:].reshape(src.shape[0], -1)
        words_src = src[:,[11,12],:].reshape(src.shape[0], -1)
        places_src = src[:,3,:].reshape(src.shape[0], -1)
        bodies_src = src[:,[4,13],:].reshape(src.shape[0], -1)
        src = self.fmri_embed(dorsal_src, ventral_src, 
                              visualrois_src, faces_src, words_src, places_src, bodies_src,
                              src)
        return src

    def positional_encoding(self, sequence, max_len):
        batch_size, seq_len, embedding_dim = sequence.size()

        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1) 
        div_term = torch.exp(torch.arange(0, embedding_dim, 2).float() * -(math.log(10000.0) / embedding_dim)) 
        
        pos_encoding = torch.zeros(max_len, embedding_dim)
        pos_encoding[:, 0::2] = torch.sin(position * div_term)
        pos_encoding[:, 1::2] = torch.cos(position * div_term)

        pos_encoding = pos_encoding.unsqueeze(0)
        pos_encoding = pos_encoding.repeat(batch_size, 1, 1)  

        pos_encoding = pos_encoding[:, :seq_len, :]
        return pos_encoding
    
    def _compute_contrastive_loss(self, token, target):
        return F.cosine_embedding_loss(token, target, torch.ones(token.size(0)).to(self.device))

    def _compute_mse_loss(self, token, target):
        return F.mse_loss(token, target)

    def _compute_ortho_loss(self, high_token, low_token, g_h, g_l):
        batch_size = high_token.size(0)
        raw_ortho_loss = torch.norm(torch.matmul(low_token.unsqueeze(2), high_token.unsqueeze(1)), p='fro') ** 2
        ortho_loss = torch.mean(g_h * g_l) * raw_ortho_loss / (batch_size ** 2 * high_token.size(-1))
        return ortho_loss

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
        return mAPvalue
    
    def forward(self, src, hf, lf):
        batch_size = src.size(0)
        src = self.multiscale_fmri_embed(src)
        ls_brain_token = self.ls_brain_token.expand(batch_size, -1, -1)
        ms_brain_token = self.ms_brain_token.expand(batch_size, -1, -1)
        ss_brain_token = self.ss_brain_token.expand(batch_size, -1, -1)
    
        high_token = self.high_feature_token.expand(batch_size, -1, -1)
        low_token = self.low_feature_token.expand(batch_size, -1, -1)

        src = torch.cat((ls_brain_token, ms_brain_token, ss_brain_token, src, high_token, low_token), dim=1)
        src *= 2 * src.shape[-1] ** 0.5
        src += self.a_fmri * self.positional_encoding(src, src.shape[1]).to(self.device)
        
        attn_weights_list = []

        memory = src
        for layer in self.encoder_layers:
            multi_scale_self_attn_output, attn_weights = layer['multiscale_self_attn'](memory.permute(1, 0, 2))
            for v in attn_weights.values():
                attn_weights_list.append(v.detach().clone())
            multi_scale_self_attn_output = multi_scale_self_attn_output.permute(1, 0, 2)
            multi_scale_self_attn_output = layer['layer1_norm1'](memory + layer['layer1_dropout'](multi_scale_self_attn_output))
            multi_scale_self_attn_output = layer['layer1_norm2'](multi_scale_self_attn_output + layer['layer1_dropout'](layer['layer1_ffn'](multi_scale_self_attn_output)))
            
            brain_token, high_token, low_token = multi_scale_self_attn_output[:,:-2,:], multi_scale_self_attn_output[:,-2,:], multi_scale_self_attn_output[:,-1,:]
            high_token, low_token, g_h, g_l = self.gated(high_token, low_token)

            cross_attn1_output, _ = layer['cross_attn1'](
                torch.cat([high_token.unsqueeze(1), low_token.unsqueeze(1)], dim=1).permute(1, 0, 2),
                brain_token.permute(1, 0, 2),
                brain_token.permute(1, 0, 2)
            )

            cross_attn1_output = cross_attn1_output.permute(1, 0, 2)
            cross_attn1_output = layer['layer2_norm1'](torch.cat([high_token.unsqueeze(1), low_token.unsqueeze(1)], dim=1) + layer['layer2_dropout'](cross_attn1_output))
            cross_attn1_output = layer['layer2_norm2'](cross_attn1_output + layer['layer2_dropout'](layer['layer2_ffn'](cross_attn1_output)))
            memory  = torch.cat([brain_token, cross_attn1_output], dim=1)

        hf_pred = self.high_fc(high_token)
        lf_pred = self.low_fc(low_token)
        # calculate loss
        contrastive_loss = self._compute_contrastive_loss(hf_pred, hf)
        mse_loss = self._compute_mse_loss(lf_pred, lf)
        ortho_loss = self._compute_ortho_loss(high_token, low_token, g_h, g_l)
        return contrastive_loss, mse_loss, ortho_loss, hf_pred, lf_pred