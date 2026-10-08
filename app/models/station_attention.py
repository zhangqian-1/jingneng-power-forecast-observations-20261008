"""StationAttentionHF 模型定义"""
from __future__ import annotations

import torch
from torch import nn


class ResidualConvBlock(nn.Module):
    """带残差连接的膨胀卷积块"""

    def __init__(self, channels: int, kernel_size: int, dilation: int, dropout: float) -> None:
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation),
            nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.net(x.transpose(1, 2)).transpose(1, 2)
        return self.norm(x + y)


class StationAttentionHF(nn.Module):
    """
    站点级Attention模型 + 高频损失优化

    输入：
        - x: 历史多变量时间序列 [batch, input_size, n_features]
        - future: 未来时间特征 [batch, horizon, n_future_features]

    输出：
        - pred: 未来预测值 [batch, horizon]
    """

    def __init__(
        self,
        n_features: int,
        n_future_features: int,
        input_size: int,
        horizon: int,
        target_feature_idx: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.horizon = horizon
        self.target_feature_idx = target_feature_idx

        # 输入投影
        self.input_proj = nn.Linear(n_features, d_model)
        self.pos = nn.Parameter(torch.zeros(1, input_size, d_model))

        # 膨胀卷积块（捕获多尺度时间模式）
        self.conv_blocks = nn.ModuleList(
            [
                ResidualConvBlock(d_model, kernel_size=5, dilation=1, dropout=dropout),
                ResidualConvBlock(d_model, kernel_size=5, dilation=2, dropout=dropout),
                ResidualConvBlock(d_model, kernel_size=5, dilation=4, dropout=dropout),
            ]
        )

        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers,
            enable_nested_tensor=False,
        )

        # Attention池化
        self.attn_pool = nn.Linear(d_model, 1)

        # 上下文投影
        self.context_proj = nn.Sequential(
            nn.Linear(d_model * 3, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # 未来特征投影
        self.future_proj = nn.Sequential(
            nn.Linear(n_future_features, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # 解码器（预测残差）
        self.decoder = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
        )

        # 残差缩放参数
        self.residual_scale = nn.Parameter(torch.tensor(0.35))

    def forward(self, x: torch.Tensor, future: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            x: 历史输入 [batch, input_size, n_features]
            future: 未来特征 [batch, horizon, n_future_features]

        Returns:
            pred: 预测值 [batch, horizon]
        """
        # 编码历史输入
        h = self.input_proj(x) + self.pos[:, : x.shape[1]]

        # 膨胀卷积
        for block in self.conv_blocks:
            h = block(h)

        # Transformer编码
        h = self.encoder(h)

        # Attention池化 + 上下文聚合
        weights = torch.softmax(self.attn_pool(h), dim=1)
        pooled = torch.sum(weights * h, dim=1)
        last = h[:, -1]
        mean = h.mean(dim=1)
        context = self.context_proj(torch.cat([last, mean, pooled], dim=-1))

        # 扩展context到horizon维度
        context = context[:, None, :].expand(-1, self.horizon, -1)

        # 未来特征投影
        fut = self.future_proj(future)

        # 解码：预测残差
        residual = self.decoder(torch.cat([context, fut], dim=-1)).squeeze(-1)

        # Baseline（前一天同时刻） + 残差
        baseline = x[:, -self.horizon :, self.target_feature_idx]
        return baseline + self.residual_scale * residual
