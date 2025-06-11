import numpy as np

from config import Config
from src.train import train_model
from src.inference import predict_user_tags, visualize_clusters
from src.utils import analyze_results


def main():
    config = Config()

    # 步骤1: 训练模型
    print("Starting model training...")
    train_model(config)

    # 步骤2: 预测用户标签
    print("\nPredicting user tags...")
    result = predict_user_tags(config)

    # 步骤3: 可视化聚类结果
    print("\nVisualizing clusters...")
    visualize_clusters(np.array(result['embedding'].tolist()),
                       result['severity'].values, config)

    # 步骤4: 分析结果
    print("\nAnalyzing results...")
    stats = analyze_results(config.CLUSTER_SAVE_PATH)
    print("\nTag Statistics:")
    print(stats)

    print("\nCompleted fee collection tagging!")


if __name__ == "__main__":
    main()
