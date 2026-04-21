# CIFAR-10 - 图像分类挑战赛

## 比赛概述

- **任务**：10分类（airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck）
- **数据**：32×32 RGB 彩色图像，训练集 50000 张，测试集 300000 张
- **评估指标**：Accuracy（准确率）
- **方法**：PyTorch CNN（InceptionA + ResidualBlock）

## 网络结构

```
Input(3,32,32)
  → Conv2d(3→16, k=5) → ReLU → MaxPool(2)        # (16,14,14)
  → ResidualBlock(16)                                # (16,14,14)
  → InceptionA(16→88)                                # (88,14,14)
  → Conv2d(88→16, k=5) → ReLU → MaxPool(2)         # (16,5,5)
  → ResidualBlock(16)                                # (16,5,5)
  → InceptionA(16→88)                                # (88,5,5)
  → Flatten → Linear(2200→10)
```

### InceptionA 模块

四个分支并行计算，最后在通道维度拼接：
- 1×1 卷积（16通道）
- 1×1 → 5×5 卷积（24通道）
- 1×1 → 3×3 → 3×3 卷积（24通道）
- 平均池化 → 1×1 卷积（24通道）
- 总输出：16 + 24 + 24 + 24 = **88 通道**

### ResidualBlock 模块

残差连接：`output = F.relu(conv2(relu(conv1(x))) + x)`
- 输入直接加到输出上，解决深层网络梯度消失问题

## 训练配置

- **损失函数**：CrossEntropyLoss
- **优化器**：Adam（lr=0.001）
- **Batch size**：64
- **Epochs**：30

## 数据预处理

- 用 PIL 读取 PNG 图片
- `.permute(0, 3, 1, 2)` 转为 PyTorch 的 NCHW 格式
- 除以 255.0 归一化到 [0, 1]
- 标签从字符串映射为 0-9 的整数索引

## 优化历程

### v1 - 基础 CNN
- 2层卷积（3→10→20）+ 全连接
- SGD 优化器，lr=0.01
- 10个 epoch
- **问题**：`predict_save` 用 `range(1, len+1)` 生成 id，与实际图片 id 顺序不匹配（`os.listdir` 返回字符串排序而非数字排序）
- **Kaggle 得分**：0.09

### v2 - 修复 id 排序 + 归一化
- 修复 test id 排序：先转 int 再 sorted
- 数据归一化（/ 255.0）
- 学习率降至 0.001，epoch 增加到 30
- **Kaggle 得分**：0.66

### v3 - 添加 InceptionA 模块
- 引入 InceptionA 多分支并行结构，捕获多尺度特征
- **Kaggle 得分**：0.67，提升不明显

### v4 - 添加 ResidualBlock + InceptionA
- ResidualBlock 帮助梯度传播，InceptionA 捕获多尺度特征
- Conv 层将 InceptionA 的 88 通道转回 16，以便复用 ResidualBlock
- **Kaggle 得分**：0.67

### v5 - 尝试数据增强（已回退）
- 添加 RandomHorizontalFlip + RandomCrop
- **结果**：提升不大，已回退

## 关键知识点

1. **os.listdir 排序问题**：`os.listdir()` 返回字符串排序（1, 10, 100...），需要转 int 后再排序才能得到正确的数字顺序
2. **数据归一化的重要性**：像素值 0-255 不归一化会导致梯度爆炸，模型几乎无法训练（从 0.09 到 0.66 的差距）
3. **InceptionA 通道计算**：四个分支的输出通道数分别为 16, 24, 24, 24，拼接后为 88
4. **ResidualBlock 通道要求**：输入输出通道数必须一致，才能做残差连接 `y + x`
5. **padding 的作用**：`kernel_size=5` 配合 `padding=2` 可以保持特征图尺寸不变
6. **CrossEntropyLoss 不需要 softmax**：网络 forward 输出 raw logits，只在 predict 时手动加 softmax
