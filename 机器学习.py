def manual_prf_cm(y_true, y_score, threshold=0.5):
    # 根据阈值将概率转换为预测标签
    y_pred = []
    for score in y_score:
        # 手动实现 (y_score >= threshold).astype(int)
        if score >= threshold:
            y_pred.append(1)
        else:
            y_pred.append(0)

    # 统计 TP, FP, TN, FN
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    # 遍历真实标签和预测标签
    for true_label, pred_label in zip(y_true, y_pred):
        if true_label == 1 and pred_label == 1:
            TP += 1
        elif true_label == 0 and pred_label == 1:
            FP += 1
        elif true_label == 0 and pred_label == 0:
            TN += 1
        elif true_label == 1 and pred_label == 0:
            FN += 1

    # 计算 P, R, F1
    if (TP + FP) == 0:
        precision = 0.0
    else:
        precision = TP / (TP + FP)

    if (TP + FN) == 0:
        recall = 0.0
    else:
        recall = TP / (TP + FN)

    if (precision + recall) == 0:
        f1 = 0.0
    else:
        f1 = 2 * (precision * recall) / (precision + recall)

    # 创建混淆矩阵
    confusion_matrix_result = [[TN, FP], [FN, TP]]

    return {
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': confusion_matrix_result,
        'y_pred': y_pred
    }


y_true = [0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 1]
y_score = [0.1, 0.9, 0.3, 0.8, 0.2, 0.1, 0.4, 0.7, 0.6, 0.3, 0.9]

print("真实标签:", y_true)
print("预测概率:", y_score)
print("=" * 50)

# 计算并显示指标 (阈值=0.5)
threshold_main = 0.5
metrics = manual_prf_cm(y_true, y_score, threshold=threshold_main)

print(f"阈值 ({threshold_main}) 结果:")
print(f"  预测标签: {metrics['y_pred']}")
print(f"  精确率 (Precision): {metrics['precision']:.4f}")
print(f"  召回率 (Recall): {metrics['recall']:.4f}")
print(f"  F1分数: {metrics['f1_score']:.4f}")
print(f"  混淆矩阵:\n  {metrics['confusion_matrix'][0]}\n  {metrics['confusion_matrix'][1]}")
print("=" * 50)

# 测试不同阈值的效果
print("\n不同阈值下的性能:")
print("阈值 | 精确率 | 召回率 | F1分数")
print("-" * 30)
for threshold in [0.3, 0.5, 0.7]:
    metrics = manual_prf_cm(y_true, y_score, threshold)
    print(f"{threshold} | {metrics['precision']:.4f} | {metrics['recall']:.4f} | {metrics['f1_score']:.4f}")
