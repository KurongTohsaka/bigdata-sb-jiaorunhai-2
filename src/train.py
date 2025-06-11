import torch
import torch.optim as optim
from tqdm import tqdm
import os

from src.model import SimCLRModel, contrastive_loss
from src.data_preprocessing import load_data, prepare_dataloader


def train_model(config):
    """训练SimCLR模型"""
    # 创建输出目录
    os.makedirs(os.path.dirname(config.MODEL_SAVE_PATH), exist_ok=True)

    # 加载数据
    features, user_ids, scaler = load_data()
    dataloader = prepare_dataloader(features, config)

    # 初始化模型
    device = config.DEVICE
    model = SimCLRModel(config).to(device)

    # 优化器
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY
    )

    # 训练循环
    best_loss = float('inf')
    for epoch in range(config.EPOCHS):
        model.train()
        total_loss = 0

        for batch in tqdm(dataloader, desc=f"Epoch {epoch + 1}/{config.EPOCHS}"):
            x, x_aug1, x_aug2 = [t.to(device) for t in batch]

            # 前向传播
            _, proj1 = model(x_aug1)
            _, proj2 = model(x_aug2)

            # 计算损失
            loss = contrastive_loss(proj1, proj2, config.TEMPERATURE)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # 计算平均损失
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch + 1}/{config.EPOCHS}, Loss: {avg_loss:.4f}")
        if avg_loss <= 0 or torch.isnan(torch.tensor(avg_loss)):
            print(f"警告：异常损失值 {avg_loss}")

        # 保存最佳模型
        if avg_loss < best_loss:
            best_loss = avg_loss
            # 确保目录存在
            os.makedirs(os.path.dirname(config.MODEL_SAVE_PATH), exist_ok=True)
            torch.save({
                'model_state_dict': model.state_dict(),
                'scaler': scaler
            }, config.MODEL_SAVE_PATH)
            print(f"Saved model with loss {best_loss:.4f}")

    return model


def extract_embeddings(model, features, config):
    """提取用户嵌入特征"""
    device = config.DEVICE
    model.eval()

    with torch.no_grad():
        features_tensor = torch.tensor(features, dtype=torch.float32).to(device)
        embeddings, _ = model(features_tensor)

    return embeddings.cpu().numpy()


def cluster_users(embeddings, config):
    """对用户嵌入进行聚类"""
    from sklearn.cluster import KMeans
    import numpy as np

    # K-means聚类
    kmeans = KMeans(
        n_clusters=config.NUM_CLUSTERS,
        random_state=42,
        n_init=10
    )
    clusters = kmeans.fit_predict(embeddings)

    # 基于聚类中心到原点的距离分配标签严重程度
    cluster_centers = kmeans.cluster_centers_
    center_distances = np.linalg.norm(cluster_centers, axis=1)

    # 根据严重程度排序集群
    severity_order = np.argsort(center_distances)[::-1]
    severity_mapping = {old_id: new_id for new_id, old_id in enumerate(severity_order)}

    # 重映射聚类标签
    severity_labels = np.array([severity_mapping[c] for c in clusters])

    return severity_labels, cluster_centers
