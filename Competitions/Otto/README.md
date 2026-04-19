# Otto Group Product Classification Challenge

## Competition Overview

- **Task**: Multi-class classification (9 classes)
- **Features**: 93 anonymized numerical features
- **Metric**: Multi-class log-loss
- **Method**: PyTorch fully-connected neural network

## Network Architecture

```
Input(93) → Linear(93,64) → ReLU → Dropout(0.3)
         → Linear(64,32)  → ReLU → Dropout(0.3)
         → Linear(32,16)  → ReLU → Dropout(0.3)
         → Linear(16,9)   → Softmax → 9-class probabilities
```

## Training Configuration

- **Loss**: CrossEntropyLoss
- **Optimizer**: Adam (lr=0.01)
- **Batch size**: 64
- **Epochs**: 100

## Optimization History

### v1 - Baseline
- 4-layer network (93→64→32→16→9)
- SGD optimizer, lr=0.01, momentum=0.5
- 50 epochs
- **Issue**: `predict` method used `torch.max` + `pd.get_dummies`, output was TRUE/FALSE instead of probabilities

### v2 - Fix prediction output
- Replaced `get_dummies` with `softmax` to output proper probabilities
- Kaggle score: ~0.53

### v3 - Add Dropout + Adam + more epochs
- Added Dropout(0.3) between hidden layers
- Switched optimizer from SGD to Adam
- Increased epochs from 50 to 100
- Added `model.train()` / `model.eval()` for proper Dropout behavior
- Score improved slightly

### v4 - Deeper network + BatchNorm + Normalization (failed)
- Added 5th layer (93→128→64→32→16→9)
- Added BatchNorm1d after each hidden layer
- Added Z-score feature normalization
- **Result**: Score dropped from 0.7x to 0.55
- **Lesson**: Multiple changes at once can conflict. Normalization changed data distribution but lr=0.01 was too large for Adam with normalized data. Deeper network + BatchNorm + Dropout over-regularized the model. Optimize one variable at a time.

## Key Takeaways

1. **CrossEntropyLoss vs Softmax**: `CrossEntropyLoss` internally applies softmax, so the network's `forward` should output raw logits. Only add softmax in `predict` for inference.
2. **Dropout requires train/eval mode**: `model.train()` enables dropout during training; `model.eval()` disables it during inference.
3. **One change at a time**: When optimizing, change one hyperparameter at a time to understand its effect. Multiple simultaneous changes make it impossible to identify what helped or hurt.
4. **Training loss ≠ test performance**: A slightly higher training loss with regularization (Dropout) can lead to better test performance.
