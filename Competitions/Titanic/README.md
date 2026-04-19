# Titanic - 机器学习从灾难中学习

## 比赛概述

- **任务**：二分类（存活 / 未存活）
- **特征**：Pclass, Sex, Age, SibSp, Parch, Fare（6个特征）
- **评估指标**：Accuracy（准确率）
- **方法**：PyTorch 全连接神经网络

## 网络结构

```
Input(6) → Linear(6,3) → Sigmoid → Dropout(0.2)
         → Linear(3,1) → Sigmoid → 二分类输出
```

## 训练配置

- **损失函数**：BCELoss
- **优化器**：SGD（lr=0.005）
- **Batch size**：16
- **Epochs**：100
- **训练集/测试集划分**：80% / 20%

## 数据预处理

- 缺失的 `Age` 值用均值填充
- `Sex` 映射为 0（male）/ 1（female）
- 选取了6个特征：Pclass, Sex, Age, SibSp, Parch, Fare

## 优化方向

1. **隐藏层激活函数**：当前隐藏层使用 Sigmoid，容易导致梯度消失，建议换成 ReLU
2. **优化器**：SGD 收敛较慢，可以尝试 Adam
3. **特征工程**：`Cabin`、`Embarked`、`Name`（提取称谓）等特征尚未使用，可能有助于提升性能
4. **特征归一化**：`Fare` 和 `Age` 等特征尺度差异较大，建议做归一化处理
