import torch

class Config:
    # Data configuration
    DATA_PATH = 'dataset/data.csv'  # 修正数据集路径
    NUM_FEATURES = 7  # 根据实际特征数量调整（原为8）
    # Device configuration
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    # Training parameters
    BATCH_SIZE = 128
    EPOCHS = 128
    LEARNING_RATE = 5e-2
    WEIGHT_DECAY = 1e-4
    TEMPERATURE = 0.3  # Contrastive loss temperature

    # Model parameters
    EMBEDDING_DIM = 128
    PROJECTION_DIM = 32
    HIDDEN_DIM = 256

    # Augmentation parameters
    JITTER_SCALE = 0.1
    DROP_OUT_RATE = 0.1

    # Cluster parameters
    NUM_CLUSTERS = 3  # 轻度提醒、中度催缴、重度追讨

    # Output paths
    MODEL_SAVE_PATH = 'checkpoints/simclr_model.pth'
    CLUSTER_SAVE_PATH = 'outputs/user_clusters.csv'
