import torch
import torch.nn as nn
import torch.nn.functional as F


class ProjectionHead(nn.Module):
    """SimCLR投影头"""

    def __init__(self, config):
        super().__init__()
        self.fc1 = nn.Linear(config.EMBEDDING_DIM, config.HIDDEN_DIM)
        self.fc2 = nn.Linear(config.HIDDEN_DIM, config.PROJECTION_DIM)
        self.bn1 = nn.BatchNorm1d(config.HIDDEN_DIM)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


class SimCLRModel(nn.Module):
    """SimCLR模型架构"""

    def __init__(self, config):
        super().__init__()
        self.config = config

        # 特征嵌入网络 - 修正输入维度以匹配7个特征
        self.embedding_net = nn.Sequential(
            nn.Linear(config.NUM_FEATURES, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Linear(128, config.EMBEDDING_DIM),
            nn.BatchNorm1d(config.EMBEDDING_DIM),
            nn.ReLU(inplace=True)
        )

        # 投影头
        self.projection = ProjectionHead(config)

    def forward(self, x):
        # 添加输入维度检查
        if x.shape[1] != self.config.NUM_FEATURES:
            raise ValueError(f"输入特征维度不匹配: 预期{self.config.NUM_FEATURES}, 实际{x.shape[1]}")

        # 特征嵌入
        embedding = self.embedding_net(x)

        # 投影（在对比学习中使用）
        projection = self.projection(embedding)
        projection = F.normalize(projection, dim=1)

        return embedding, projection


def contrastive_loss(projection1, projection2, temperature):
    """NT-Xent损失函数"""
    n = projection1.size(0)
    device = projection1.device

    # 合并投影
    projections = torch.cat([projection1, projection2], dim=0)

    # 计算相似度矩阵
    similarity_matrix = F.cosine_similarity(
        projections.unsqueeze(1),
        projections.unsqueeze(0),
        dim=2
    ) / temperature

    # 创建正样本掩码：每个样本i与i+n是正样本对
    mask = torch.zeros(2*n, 2*n, dtype=torch.bool, device=device)
    mask[:n, n:] = torch.eye(n, device=device)  # 上右块对角线为True
    mask[n:, :n] = torch.eye(n, device=device)  # 下左块对角线为True

    # 提取正样本相似度并重塑
    pos_similarities = similarity_matrix[mask].view(2*n, 1)

    # 计算分母（排除自身相似度）
    diag_mask = ~torch.eye(2*n, dtype=torch.bool, device=device)
    denom = similarity_matrix[diag_mask].view(2*n, -1).exp().sum(dim=1, keepdim=True)

    # 计算损失
    loss = -torch.log(pos_similarities.exp() / denom).mean()

    return loss
