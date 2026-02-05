# Motion Transformer (MTR): https://arxiv.org/abs/2209.13508
# Published at NeurIPS 2022
# Written by Shaoshuai Shi 
# All Rights Reserved


import numpy as np
import torch
import torch.nn as nn
from ..utils import common_layers
from models.utils.layers.Mamba import Mamba
from models.utils.layers.Mamba import Encoder,EncoderLayer,EncoderLayerV2
from models.utils.Experts import TSTemporalSpatialModule
from models.utils.Experts import STTemporalSpatialModule
from models.utils.Experts import TTTemporalModule
from models.utils.Experts import SSSpatialModule
import torch.nn.functional as F


class SharedMambaBlocks(nn.Module):
    """共享时空Mamba模块的基类"""
    def __init__(self, mid_feature, num_kpt, depth=1):
        super().__init__()
        # 构建共享的时空编码器
        self.shared_temporal = self._build_encoder(d_model=num_kpt*2, depth=depth)
        self.shared_spatial = self._build_encoder(d_model=mid_feature, depth=depth)
        
    def _build_encoder(self, d_model, depth):
        return Encoder(
            [
                EncoderLayer(
                    Mamba(
                        d_model=d_model,
                        d_state=32,
                        d_conv=2,
                        expand=1
                    ),
                    Mamba(
                        d_model=d_model,
                        d_state=32,
                        d_conv=2,
                        expand=1
                    ),
                    d_model=d_model,
                    dropout=0.1,
                    activation='gelu'
                ) for _ in range(depth)
            ],
            norm_layer=nn.LayerNorm(d_model)
        )

class FourExpertsPool_Shared(SharedMambaBlocks):
    def __init__(self, mid_feature, num_kpt, depth=1, num_experts=4):
        super().__init__(mid_feature, num_kpt, depth)
        
        # 初始化专家模块（共享核心参数）
        self.ts_expert = TSTemporalSpatialModule(
            temporal_mamba=self.shared_temporal,
            spatial_mamba=self.shared_spatial
        )
        
        self.st_expert = STTemporalSpatialModule(
            spatial_mamba=self.shared_spatial,
            temporal_mamba=self.shared_temporal
        )
        
        self.tt_expert = TTTemporalModule(
            temporal_mamba1=self.shared_temporal,
            temporal_mamba2=self.shared_temporal
        )
        
        self.ss_expert = SSSpatialModule(
            spatial_mamba1=self.shared_spatial,
            spatial_mamba2=self.shared_spatial
        )

        self.experts = nn.ModuleList([
            self.ts_expert, self.st_expert,
            self.ss_expert, self.tt_expert
        ])

    def forward(self, x):
        expert_outputs = [expert(x) for expert in self.experts]
        return torch.stack(expert_outputs, dim=1)
    
class ExpertGating(nn.Module):
    def __init__(self, input_dim, num_experts=4, top_k=2):
        super().__init__()
        self.top_k = top_k
        
        # 门控生成器
        self.gate_generator = nn.Linear(input_dim, num_experts)
        self.capture_gating_weights = False  # 新增捕获开关
        self.captured_weights = None         # 存储捕获的权重

        # 初始化参数
        nn.init.xavier_normal_(self.gate_generator.weight)
        nn.init.constant_(self.gate_generator.bias, 0.1)

    def forward(self, x):
        # 输入形状: [B, J3, T]
        B, J3, T = x.shape
        
        # 时空全局平均池化
        pooled = x.mean(dim=[-2])  # [B,seq_len]
        
        # 生成门控权重
        logits = self.gate_generator(pooled)  # [B, 4]

        if self.capture_gating_weights:
            # 捕获原始权重值（非稀疏）
            full_weights = F.softmax(logits, dim=-1)
            self.captured_weights = full_weights.clone().detach()
        
        # Top-K选择
        top_logits, indices = torch.topk(logits, self.top_k, dim=-1)
        sparse_weights = torch.zeros_like(logits).scatter(-1, indices, F.softmax(top_logits, dim=-1))
        
        return sparse_weights, indices

class MoELayer(nn.Module):
    def __init__(self, mid_feature, num_kpt):
        super().__init__()
        self.expert_pool = FourExpertsPool_Shared(
            mid_feature=mid_feature,
            num_kpt=num_kpt,
            depth=3,
            num_experts=4
        )
        self.top_k = 1
        # self.gate = ExpertGating(
        #     input_dim=mid_feature,
        #     num_experts=4,
        #     top_k = 1
        # )

        self.gate = ExpertGating(
            input_dim=mid_feature,
            num_experts=4,
            top_k=1
        )
        # 添加标记用于捕获专家输出
        self.capture_expert_output = False
        self.expert_outputs = None  # 用于存储专家输出
         # 残差系数（可学习）
        # self.res_coef = nn.Parameter(torch.tensor(0.1))
        # self.norm = nn.LayerNorm(mid_feature)

    def forward(self, x):
        # residual = x  # [B, J3, T]
        B, J3, T = x.shape
         # 标准化
        # x = self.norm(x)
        # 获取所有专家输出
        expert_outputs = self.expert_pool(x)  # [B, 4, J3, T]

        # 如果设置了捕获标记，保存专家输出
        if self.capture_expert_output:
            # 平均池化减少序列长度维度
            self.expert_outputs = expert_outputs.mean(dim=-1)  # [B, 4, J3]
            # 展平空间维度
            self.expert_outputs = self.expert_outputs.reshape(
                self.expert_outputs.shape[0], 
                self.expert_outputs.shape[1], 
                -1
            )  # [B, 4, J3] -> [B, 4, num_kpt*2]
            
        # 获取门控权重
        gate_weights, indices = self.gate(x)  # [B, 4], [B, 2]
        
         # 动态选择专家
        selected_experts = torch.gather(
            expert_outputs,
            dim=1,
            index=indices.view(B, self.top_k, 1, 1).expand(-1, -1, J3, T)
        )  # [B, 2, J3, T]
        
        # 加权融合
        weights = gate_weights.gather(1, indices).view(B, self.top_k, 1, 1)  # [B, 2, 1, 1]
        moe_out = (selected_experts * weights).sum(dim=1)  # [B, J3, T]
        
        # 残差连接
        # return moe_out + self.res_coef * residual
        return moe_out

class PointNetPolylineEncoder(nn.Module):
    def __init__(self, in_channels, hidden_dim, num_layers=3, num_pre_layers=2, out_channels=None, num_kpt=3, mid_feature = 10):
        super().__init__()
        self.num_kpt = num_kpt
        self.dct, self.idct = self.get_dct_matrix(self.num_kpt*2)
        self.mid_feature = mid_feature
        self.pre_mlps = common_layers.build_mlps(
            c_in=in_channels,
            mlp_channels=[hidden_dim] * num_pre_layers,
            ret_before_act=False
        )
        self.mlps = common_layers.build_mlps(
            c_in=hidden_dim * 2,
            mlp_channels=[hidden_dim] * (num_layers - num_pre_layers),
            ret_before_act=False
        )
        # 初始化MoE层
        self.moe_layers = nn.ModuleList([
            MoELayer(
                mid_feature=self.mid_feature,
                num_kpt=num_kpt,
            ) for _ in range(1)
        ])
        
        if out_channels is not None:
            self.out_mlps = common_layers.build_mlps(
                c_in=hidden_dim, mlp_channels=[hidden_dim, out_channels], 
                ret_before_act=True, without_norm=True
            )
        else:
            self.out_mlps = None 

    def forward(self, polylines, polylines_mask):
        """
        Args:
            polylines (batch_size, num_polylines, num_points_each_polylines, C):
            polylines_mask (batch_size, num_polylines, num_points_each_polylines):

        Returns:
        """
        transpose_polylines = polylines.permute(0, 1, 3, 2)  # [B, A, C, Tp]
        input = torch.matmul(self.dct, transpose_polylines)  # [B, A, C, Tp]

        batch_size, num_polylines,  num_points_each_polylines, C = polylines.shape

        for i in range(num_polylines):

            people_feature = input[:, i, :, :] # [B, C, Tp]
            
            filter_feature = people_feature.clone() # [B, C, Tp]
            
            # 通过MoE层处理
            for moe_layer in self.moe_layers:
                filter_feature = moe_layer(filter_feature)  # [B, C, Tp]

            feature = filter_feature + people_feature.clone() # [B, C, Tp]
            feature = torch.matmul(self.idct, feature) # [B, C, Tp]
            feature = feature.transpose(1, 2) # [B, Tp, C]

            if i == 0:
                polylines_pre = feature.unsqueeze(1).clone()
            else:
                polylines_pre = torch.cat([polylines_pre, feature.unsqueeze(1).clone()], 1) # [B, A, Tp, C]

        # pre-mlp
        polylines_feature_valid = self.pre_mlps(polylines_pre[polylines_mask])  # (N, C)
        polylines_feature = polylines_pre.new_zeros(batch_size, num_polylines,  num_points_each_polylines, polylines_feature_valid.shape[-1])
        polylines_feature[polylines_mask] = polylines_feature_valid

        # get global feature
        pooled_feature = polylines_feature.max(dim=2)[0]
        polylines_feature = torch.cat((polylines_feature, pooled_feature[:, :, None, :].repeat(1, 1, num_points_each_polylines, 1)), dim=-1)

        # mlp
        polylines_feature_valid = self.mlps(polylines_feature[polylines_mask])
        feature_buffers = polylines_feature.new_zeros(batch_size, num_polylines, num_points_each_polylines, polylines_feature_valid.shape[-1])
        feature_buffers[polylines_mask] = polylines_feature_valid

        # max-pooling
        feature_buffers = feature_buffers.max(dim=2)[0]  # (batch_size, num_polylines, C)
        
        # out-mlp 
        if self.out_mlps is not None:
            valid_mask = (polylines_mask.sum(dim=-1) > 0)
            feature_buffers_valid = self.out_mlps(feature_buffers[valid_mask])  # (N, C)
            feature_buffers = feature_buffers.new_zeros(batch_size, num_polylines, feature_buffers_valid.shape[-1])
            feature_buffers[valid_mask] = feature_buffers_valid
        return feature_buffers
    
    def get_dct_matrix(self, N):
        # Computes the discrete cosine transform (DCT) matrix and its inverse (IDCT)
        dct_m = np.eye(N)
        for k in np.arange(N):
            for i in np.arange(N):
                w = np.sqrt(2 / N)
                if k == 0:
                    w = np.sqrt(1 / N)
                dct_m[k, i] = w * np.cos(np.pi * (i + 1 / 2) * k / N)
        idct_m = np.linalg.inv(dct_m)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dct_m = torch.tensor(dct_m).float().to(device)
        idct_m = torch.tensor(idct_m).float().to(device)
        return dct_m, idct_m
