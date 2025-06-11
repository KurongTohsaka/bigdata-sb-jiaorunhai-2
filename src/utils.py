import os
import pandas as pd
import matplotlib.pyplot as plt


def analyze_results(result_path):
    """分析聚类结果"""
    # 检查文件是否存在
    if not os.path.exists(result_path):
        raise FileNotFoundError(f"结果文件不存在: {result_path}")

    df = pd.read_csv(result_path)

    # 创建输出目录
    os.makedirs('outputs', exist_ok=True)

    # 计算每个标签的统计信息
    tag_stats = df.groupby('tag').agg({
        'user_id': 'count',
        'severity': ['mean', 'std']
    })

    tag_stats.columns = ['count', 'severity_mean', 'severity_std']
    tag_stats['percentage'] = tag_stats['count'] / len(df) * 100

    # 绘制标签分布
    plt.figure(figsize=(10, 6))
    ax = tag_stats['count'].sort_values().plot(kind='barh', color='skyblue')
    plt.title('Distribution of Fee Collection Tags')
    plt.xlabel('Number of Users')
    plt.ylabel('Tag')

    # 添加数量标签
    for i, v in enumerate(tag_stats['count'].sort_values()):
        ax.text(v + 3, i - 0.1, str(v), color='black', fontweight='bold')

    plt.tight_layout()
    plt.savefig('outputs/tag_distribution.png', dpi=300)

    return tag_stats
