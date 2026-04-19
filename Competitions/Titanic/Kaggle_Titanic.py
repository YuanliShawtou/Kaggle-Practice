import numpy as np
import pandas as pd
from torch.jit import export_opnames
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import torch

class TitanicDataset(Dataset):
    def __init__(self, filePath):
        features = ["Pclass",'Sex','Age',"SibSp", "Parch", "Fare"]
        data = pd.read_csv(filePath)
        data = data.fillna(data['Age'].mean())
        self.len = data.shape[0]

        self.x_data = torch.from_numpy(np.array(data[features]))
        self.y_data = torch.from_numpy(np.array(data['Survived']))

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return  self.len

dataset = TitanicDataset("train.csv")

train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

train_loader = DataLoader(dataset=train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=16, shuffle=False)

class Model(torch.nn.Module):
    def __init__(self):
        super(Model,self).__init__()
        self.linear1 = torch.nn.Linear(6,3)
        self.dropout = torch.nn.Dropout(0.2)
        self.linear2 = torch.nn.Linear(3,1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self,x):
        x = self.sigmoid(self.linear1(x))
        x = self.dropout(x)
        x = self.sigmoid(self.linear2(x))
        return x

    def predict(self,x):
        with torch.no_grad():
            x = self.sigmoid(self.linear1(x))
            x = self.sigmoid(self.linear2(x))
            y = []
            for i in x:
                if i > 0.5:
                    y.append(1)
                else:
                    y.append(0)
            return y

model = Model()
criterion = torch.nn.BCELoss(reduction='mean')
optimizer = torch.optim.SGD(model.parameters(),lr=0.005)

def compute_accuracy(model, loader):
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.float()
            labels = labels.float()
            outputs = model(inputs).squeeze(-1)
            preds = (outputs > 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total

if __name__=="__main__":
    train_acc_list = []
    test_acc_list = []

    for epoch in range(100):
        for i, data in enumerate(train_loader,0):
            inputs, labels = data

            inputs = inputs.float()
            labels = labels.float()

            # forward
            y_pred = model(inputs)
            y_pred = y_pred.squeeze(-1)

            # loss
            loss = criterion(y_pred, labels)
            print(epoch,i,loss.item())

            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        train_acc = compute_accuracy(model, train_loader)
        test_acc = compute_accuracy(model, test_loader)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print(f"Epoch {epoch}: Train Acc={train_acc:.4f}, Test Acc={test_acc:.4f}")

    plt.plot(range(100), train_acc_list, label='Train Accuracy')
    plt.plot(range(100), test_acc_list, label='Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training vs Testing Accuracy')
    plt.legend()
    plt.show()

test_data = pd.read_csv("test.csv")
test_data = test_data.fillna(test_data['Age'].mean())
test_data['Sex'] = test_data['Sex'].map({'male': 0, 'female': 1})
features = ["Pclass", 'Sex',"Age", "SibSp", "Parch", "Fare"]
test = torch.from_numpy(np.array(test_data[features]))

y = model.predict(test.float())
out = pd.DataFrame({'PassengerId':test_data.PassengerId,'Survived':y})
print(out['Survived'].sum())#存活人数
out.to_csv('my_first_predict.csv',index=False)