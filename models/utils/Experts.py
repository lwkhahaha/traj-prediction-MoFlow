import numpy as np
import torch
import torch.nn as nn
from models.utils.layers.Mamba import Mamba
from models.utils.layers.Mamba import Encoder,EncoderLayer,EncoderLayerV2

class STTemporalSpatialModule(nn.Module):
    def __init__(self, spatial_mamba=None, temporal_mamba=None, seq_len=10, dropout=0.0):
        """
        Linear 版 ST 分支模块（可直接替换原 STTemporalSpatialModule）。

        说明：
        - 不再使用 spatial_mamba / temporal_mamba（为了兼容旧调用签名仍保留参数）
        - 输入/输出都为 [b, J3, seq_len]
        - 在时间维 seq_len 上做两层 Linear（相当于轻量 temporal mixing）

        Args:
            spatial_mamba (nn.Module, optional): 为兼容保留，不使用
            temporal_mamba (nn.Module, optional): 为兼容保留，不使用
            seq_len (int): 序列长度（你的 Tp / past_frames / num_kpt? 对应的时间长度），例如 10
            dropout (float): 可选 dropout
        """
        super(STTemporalSpatialModule, self).__init__()
        self.seq_len = seq_len

        self.temporal_mlp = nn.Sequential(
            nn.Linear(seq_len, seq_len),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(seq_len, seq_len),
        )

    def forward(self, filter_feature):
        """
        Args:
            filter_feature: [b, J3, seq_len]

        Returns:
            out: [b, J3, seq_len]
        """
        b, j3, t = filter_feature.shape
        if t != self.seq_len:
            raise ValueError(f"STTemporalSpatialModule (Linear) expected seq_len={self.seq_len}, but got t={t}")

        # 对每个通道的时间序列做线性混合
        out = self.temporal_mlp(filter_feature)  # Linear 默认作用在最后一维 seq_len
        return out

class TSTemporalSpatialModule(nn.Module):
    def __init__(self, temporal_mamba, spatial_mamba):
        """
        初始化 TS 分支模块。

        Args:
            temporal_mamba (nn.Module): 时间维度的 Mamba 模块。
            spatial_mamba (nn.Module): 空间维度的 Mamba 模块。
        """
        super(TSTemporalSpatialModule, self).__init__()
        self.temporal_mamba = temporal_mamba
        self.spatial_mamba = spatial_mamba

    def forward(self, filter_feature):
        """
        前向传播方法。

        Args:
            filter_feature (torch.Tensor): 输入特征，形状为 [b, J3, seq_len]。

        Returns:
            torch.Tensor: 处理后的特征，形状为 [b, J3, seq_len]。
        """
        # [b, J3, seq_len]->[b, seq_len, J3]
        temoral_feature_att, attns = self.temporal_mamba(filter_feature.transpose(1, 2))
        # [b, seq_len, J3]->[b, J3, seq_len]
        temporal_spatial_feature_att, attns = self.spatial_mamba(temoral_feature_att.transpose(1, 2))
        return temporal_spatial_feature_att
    
class TTTemporalModule(nn.Module):
    def __init__(self, temporal_mamba1, temporal_mamba2):
        """
        初始化 TT 分支模块（时间-时间）
        
        Args:
            temporal_mamba1 (nn.Module): 第一个时间维度Mamba模块
            temporal_mamba2 (nn.Module): 第二个时间维度Mamba模块
        """
        super(TTTemporalModule, self).__init__()
        self.temporal_mamba1 = temporal_mamba1
        self.temporal_mamba2 = temporal_mamba2

    def forward(self, filter_feature):
        """
        输入特征形状: [b, J3, seq_len]
        输出特征形状: [b, J3, seq_len]
        """
        # 第一次时间处理
        # [b, J3, seq_len] -> [b, seq_len, J3]
        temporal_feat, _ = self.temporal_mamba1(filter_feature.transpose(1, 2).contiguous())
        
        # 第二次时间处理（保持时间维度在中间）
        # [b, seq_len, J3] -> [b, seq_len, J3]
        temporal_feat, _ = self.temporal_mamba2(temporal_feat)
        
        # 恢复原始维度
        # [b, seq_len, J3] -> [b, J3, seq_len]
        return temporal_feat.transpose(1, 2).contiguous()
    
class SSSpatialModule(nn.Module):
    def __init__(self, spatial_mamba1, spatial_mamba2):
        """
        初始化 SS 分支模块（空间-空间）
        
        Args:
            spatial_mamba1 (nn.Module): 第一个空间维度Mamba模块
            spatial_mamba2 (nn.Module): 第二个空间维度Mamba模块
        """
        super(SSSpatialModule, self).__init__()
        self.spatial_mamba1 = spatial_mamba1
        self.spatial_mamba2 = spatial_mamba2

    def forward(self, filter_feature):
        """
        输入特征形状: [b, J3, seq_len]
        输出特征形状: [b, J3, seq_len]
        """
        # 第一次空间处理
        # [b, J3, seq_len] -> [b, J3, seq_len]
        spatial_feat, _ = self.spatial_mamba1(filter_feature)
        
        # 第二次空间处理
        # [b, J3, seq_len] -> [b, J3, seq_len]
        spatial_feat, _ = self.spatial_mamba2(spatial_feat)
        return spatial_feat
