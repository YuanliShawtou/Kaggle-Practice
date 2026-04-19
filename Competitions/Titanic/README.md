# Titanic - Machine Learning from Disaster

## Competition Overview

- **Task**: Binary classification (Survived / Not Survived)
- **Features**: Pclass, Sex, Age, SibSp, Parch, Fare (6 features)
- **Metric**: Accuracy
- **Method**: PyTorch fully-connected neural network

## Network Architecture

```
Input(6) → Linear(6,3) → Sigmoid → Dropout(0.2)
         → Linear(3,1) → Sigmoid → Binary output
```

## Training Configuration

- **Loss**: BCELoss
- **Optimizer**: SGD (lr=0.005)
- **Batch size**: 16
- **Epochs**: 100
- **Train/Test split**: 80% / 20%

## Data Preprocessing

- Missing `Age` values filled with mean
- `Sex` mapped to 0 (male) / 1 (female)
- Selected 6 features: Pclass, Sex, Age, SibSp, Parch, Fare

## Key Observations

1. **Sigmoid as hidden activation**: Using Sigmoid in hidden layers can lead to vanishing gradients. ReLU is generally preferred for hidden layers.
2. **SGD with small learning rate**: Convergence is slow; Adam optimizer could speed up training.
3. **Feature engineering opportunity**: Features like `Cabin`, `Embarked`, `Name` (title extraction) are not used but could improve performance.
4. **No feature normalization**: Features like `Fare` and `Age` have very different scales, which can affect training.
