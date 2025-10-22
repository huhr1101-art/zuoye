import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, roc_curve, confusion_matrix

def calculate_prf_auc(y_true, y_score, threshold=0.5):
    # 根据阈值将概率转换为预测标签
    y_pred = (y_score >= threshold).astype(int)

    # 计算P, R, F1
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    # 计算AUC
    auc = roc_auc_score(y_true, y_score)

    # 创建混淆矩阵
    cm = confusion_matrix(y_true, y_pred)

    return {
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'auc': auc,
        'confusion_matrix': cm,
        'y_pred': y_pred
    }


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
print(f"混淆矩阵:\n{metrics['confusion_matrix']}")
print(f"预测标签: {metrics['y_pred']}")
print("=" * 50)

# 测试不同阈值的效果
print("\n不同阈值下的性能:")
print("阈值 | 精确率 | 召回率 | F1分数")
print("-" * 30)
for threshold in [0.3, 0.5, 0.7]:
    metrics = calculate_prf_auc(y_true, y_score, threshold)
    print(f"{threshold}   | {metrics['precision']:.4f} | {metrics['recall']:.4f} | {metrics['f1_score']:.4f}")
