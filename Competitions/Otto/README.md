# Otto Group 产品分类挑战赛

## 比赛概述

- **任务**：多分类（9个类别）
- **特征**：93个匿名数值特征
- **评估指标**：Multi-class log-loss
- **方法**：PyTorch 全连接神经网络

## 网络结构

```
Input(93) → Linear(93,64) → ReLU → Dropout(0.3)
          → Linear(64,32)  → ReLU → Dropout(0.3)
          → Linear(32,16)  → ReLU → Dropout(0.3)
          → Linear(16,9)   → Softmax → 9类概率输出
```

## 训练配置

- **损失函数**：CrossEntropyLoss
- **优化器**：Adam（lr=0.01）
- **Batch size**：64
- **Epochs**：100

## 优化历程

### v1 - 基线版本
- 4层网络（93→64→32→16→9）
- SGD 优化器，lr=0.01，momentum=0.5
- 50个 epoch
- **问题**：`predict` 方法使用了 `torch.max` + `pd.get_dummies`，输出的是 TRUE/FALSE 而不是概率值

### v2 - 修复预测输出
- 将 `get_dummies` 替换为 `softmax`，正确输出概率值
- Kaggle 得分：约 0.53

### v3 - 添加 Dropout + Adam + 增加 epoch
- 在隐藏层之间添加了 Dropout(0.3)
- 优化器从 SGD 换为 Adam
- Epoch 从 50 增加到 100
- 添加了 `model.train()` / `model.eval()` 以正确控制 Dropout 行为
- 得分略有提升

### v4 - 加深网络 + BatchNorm + 归一化（失败）
- 增加第5层（93→128→64→32→16→9）
- 每个隐藏层后添加 BatchNorm1d
- 添加了 Z-score 特征归一化
- **结果**：得分从 0.7x 下降到 0.55
- **教训**：多个改动同时叠加会互相冲突。归一化改变了数据分布但 lr=0.01 对 Adam 来说太大了；更深的网络 + BatchNorm + Dropout 正则化过度，导致欠拟合。优化应该每次只改一个变量。

## 关键知识点

1. **CrossEntropyLoss 与 Softmax 的关系**：`CrossEntropyLoss` 内部已经包含了 softmax，所以网络的 `forward` 应该输出 raw logits，只在 `predict` 推理时才手动加 softmax
2. **Dropout 需要切换训练/评估模式**：`model.train()` 在训练时启用 Dropout；`model.eval()` 在推理时关闭 Dropout
3. **每次只改一个变量**：优化时一次只调整一个超参数，才能判断哪个改动有效、哪个有害
4. **训练 loss 不等于测试性能**：加入 Dropout 等正则化后训练 loss 可能略高，但测试性能可能更好
