import os

import torch
import pandas as pd

from src.model import SimCLRModel
from src.data_preprocessing import load_data


def load_trained_model(config):
    """加载训练好的模型和缩放器"""
    device = config.DEVICE
    checkpoint = torch.load(config.MODEL_SAVE_PATH, map_location=device, weights_only=False)

    model = SimCLRModel(config).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    scaler = checkpoint['scaler']

    return model, scaler


def predict_user_tags(config):
    """预测用户标签并保存结果"""
    # 创建输出目录
    os.makedirs(os.path.dirname(config.CLUSTER_SAVE_PATH), exist_ok=True)

    # 加载数据
    features, user_ids, _ = load_data()

    # 加载模型
    model, scaler = load_trained_model(config)

    # 提取嵌入特征
    with torch.no_grad():
        device = torch.device("mps")
        features_tensor = torch.tensor(features, dtype=torch.float32).to(device)
        embeddings, _ = model(features_tensor)
        embeddings = embeddings.cpu().numpy()

    # 聚类并分配标签
    from src.train import cluster_users
    cluster_labels, _ = cluster_users(embeddings, config)

    # 将严重程度转换为标签
    severity_to_tag = {
        0: '轻度提醒',
        1: '中度催缴',
        2: '重度追讨'
    }

    tags = [severity_to_tag.get(s, '未知') for s in cluster_labels]

    # 创建结果DataFrame
    result = pd.DataFrame({
        'user_id': user_ids,
        'embedding': list(embeddings),
        'severity': cluster_labels,
        'tag': tags
    })

    # 保存结果
    result.to_csv(config.CLUSTER_SAVE_PATH, index=False)
    print(f"Saved clustering results to {config.CLUSTER_SAVE_PATH}")

    return result


def visualize_clusters(embeddings, labels, config):
    """可视化聚类结果"""
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA

    # 创建输出目录
    os.makedirs('outputs', exist_ok=True)

    # 如果嵌入维度大于2，使用PCA降维
    if embeddings.shape[1] > 2:
        pca = PCA(n_components=2)
        embeddings_2d = pca.fit_transform(embeddings)
        print(f"Explained variance: {pca.explained_variance_ratio_.sum():.2f}")
    else:
        embeddings_2d = embeddings

    # 绘制散点图
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1],
                          c=labels, cmap='viridis', alpha=0.6)
    plt.colorbar(scatter, label='Severity Level')
    plt.title('User Clustering by Payment Behavior')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.grid(alpha=0.3)

    # 保存图像
    plt.savefig('outputs/cluster_visualization.png', dpi=300, bbox_inches='tight')
    plt.show()
