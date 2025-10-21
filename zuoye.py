import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, roc_curve, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import font_manager

font = font_manager.FontProperties(fname='C:/Windows/Fonts/simhei.ttf')  # Windows下的黑体路径
plt.rcParams['font.family'] = font.get_name()  # 设置matplotlib默认字体
plt.rcParams['axes.unicode_minus'] = False


def calculate_prf_auc(y_true, y_score, threshold=0.5):
    # 根据阈值将概率转换为预测标签
    y_pred = (y_score >= threshold).astype(int)

    # 计算P, R, F1
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    # 计算AUC
    auc = roc_auc_score(y_true, y_score)

    # 计算ROC曲线
    fpr, tpr, thresholds = roc_curve(y_true, y_score)

    # 创建混淆矩阵
    cm = confusion_matrix(y_true, y_pred)

    return {
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'auc': auc,
        'fpr': fpr,
        'tpr': tpr,
        'confusion_matrix': cm,
        'y_pred': y_pred
    }


def plot_performance(y_true, y_score, threshold=0.5):
    # 计算指标
    metrics = calculate_prf_auc(y_true, y_score, threshold)

    # 创建可视化图表
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # 绘制ROC曲线
    ax1.plot(metrics['fpr'], metrics['tpr'], color='darkorange', lw=2,
             label=f'ROC曲线 (AUC = {metrics["auc"]:.4f})')
    ax1.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', alpha=0.5)
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.05])
    ax1.set_xlabel('假正例率 (False Positive Rate)', fontproperties=font)
    ax1.set_ylabel('真正例率 (True Positive Rate)', fontproperties=font)
    ax1.set_title('ROC曲线', fontproperties=font)
    ax1.legend(loc='lower right')
    ax1.grid(True, alpha=0.3)

    # 绘制混淆矩阵热力图
    sns.heatmap(metrics['confusion_matrix'], annot=True, fmt='d', cmap='Blues',
                xticklabels=['负类', '正类'], yticklabels=['负类', '正类'], ax=ax2)
    ax2.set_xlabel('预测标签', fontproperties=font)
    ax2.set_ylabel('真实标签', fontproperties=font)
    ax2.set_title('混淆矩阵', fontproperties=font)

    plt.tight_layout()
    plt.show()

    return metrics


# 示例数据
y_true = np.array([0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 1])  # 真实标签
y_score = np.array([0.1, 0.9, 0.3, 0.8, 0.2, 0.1, 0.4, 0.7, 0.6, 0.3, 0.9])  # 预测概率

print("真实标签:", y_true)
print("预测概率:", y_score)
print("=" * 50)

# 计算并显示指标
metrics = calculate_prf_auc(y_true, y_score, threshold=0.5)

print(f"精确率 (Precision): {metrics['precision']:.4f}")
print(f"召回率 (Recall): {metrics['recall']:.4f}")
print(f"F1分数: {metrics['f1_score']:.4f}")
print(f"AUC: {metrics['auc']:.4f}")
print(f"预测标签: {metrics['y_pred']}")
print("=" * 50)

# 可视化结果
plot_performance(y_true, y_score, threshold=0.5)

# 测试不同阈值的效果
print("\n不同阈值下的性能:")
print("阈值 | 精确率 | 召回率 | F1分数")
print("-" * 30)
for threshold in [0.3, 0.5, 0.7]:
    metrics = calculate_prf_auc(y_true, y_score, threshold)
    print(f"{threshold}   | {metrics['precision']:.4f} | {metrics['recall']:.4f} | {metrics['f1_score']:.4f}")
