import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

from config import Config


def calculate_trend(series):
    """计算欠费趋势（简单线性回归斜率）"""
    arrears = series[series > 0]
    if len(arrears) < 2:
        return 0  # 数据不足时趋势为0
    x = np.arange(len(arrears)).reshape(-1, 1)
    y = arrears.values.reshape(-1, 1)
    model = LinearRegression()
    model.fit(x, y)
    return model.coef_[0][0]


def load_data():
    """加载原始数据并生成特征"""
    # 加载原始数据
    df = pd.read_csv(Config.DATA_PATH)

    # 处理缺失值
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
    df['行业分类'] = df['行业分类'].fillna(df['行业分类'].mode()[0])

    # 转换年月为日期格式并排序
    df['年月'] = pd.to_datetime(df['年月'], format='%Y%m')
    df = df.sort_values(['用户编号', '年月'])

    # 按用户编号分组，计算特征
    user_groups = df.groupby('用户编号')

    features_df = user_groups.agg(
        total_arrears=('欠费金额', 'sum'),
        avg_arrears=('欠费金额', 'mean'),
        max_arrears=('欠费金额', 'max'),
        late_fee_total=('滞纳金', 'sum'),
        arrears_frequency=('欠费金额', lambda x: (x > 0).sum()),
        recent_arrears=('欠费金额', lambda x: x[x > 0].iloc[-1] if not x[x > 0].empty else 0),
        arrears_trend=('欠费金额', calculate_trend)
    ).reset_index()

    # 重命名用户编号为user_id
    features_df.rename(columns={'用户编号': 'user_id'}, inplace=True)

    # 确保所有必要的特征列存在
    required_columns = ['user_id', 'total_arrears', 'avg_arrears', 'max_arrears',
                        'late_fee_total', 'arrears_frequency', 'recent_arrears',
                        'arrears_trend']
    for col in required_columns:
        if col not in features_df.columns:
            raise ValueError(f"Missing required feature column: {col}")

    # 数值特征列
    feature_cols = ['total_arrears', 'avg_arrears', 'max_arrears',
                    'late_fee_total', 'arrears_frequency', 'recent_arrears',
                    'arrears_trend']

    # 特征缩放
    scaler = StandardScaler()
    features_df[feature_cols] = scaler.fit_transform(features_df[feature_cols])

    # 转换为numpy数组
    features = features_df[feature_cols].values.astype(np.float32)
    user_ids = features_df['user_id'].values

    return features, user_ids, scaler


def create_pairs(features, config):
    """创建用于对比学习的正样本对"""
    # 应用轻微数据增强创建正样本对
    jitter = np.random.normal(0, config.JITTER_SCALE, features.shape)
    dropout_mask = np.random.binomial(1, 1 - config.DROP_OUT_RATE, features.shape)

    # 应用增强
    features_aug1 = features + jitter * np.std(features, axis=0)
    features_aug2 = (features * dropout_mask) + jitter * np.std(features, axis=0)

    # 随机打乱顺序创建负样本
    idx_shuffle = np.random.permutation(len(features))
    features_neg = features[idx_shuffle]

    return features_aug1, features_aug2, features_neg


def prepare_dataloader(features, config):
    """准备PyTorch数据加载器"""
    from torch.utils.data import DataLoader, TensorDataset
    import torch

    # 应用数据增强
    features_aug1, features_aug2, _ = create_pairs(features, config)

    # 创建TensorDataset
    dataset = TensorDataset(
        torch.tensor(features, dtype=torch.float32),
        torch.tensor(features_aug1, dtype=torch.float32),
        torch.tensor(features_aug2, dtype=torch.float32)
    )

    # 创建DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    return dataloader
