import pandas as pd
import numpy as np
import torch
from mpmath.identification import transforms
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.optim as optim

def convert(labels):
    ID = []
    target_labels = ['Class_1','Class_2','Class_3','Class_4','Class_5','Class_6','Class_7','Class_8','Class_9']
    for label in labels:
        ID.append(target_labels.index(label))
    return ID

class OttoDataset(Dataset):
    def __init__(self, csv_file):
        data = pd.read_csv(csv_file)
        labels = data['target']
        self.len = data.shape[0]

        self.x_data = torch.tensor(np.array(data)[:, 1:-1].astype(float))
        self.y_data = convert(labels)

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len

batch_size = 64

train_dataset = OttoDataset('./train.csv')
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear1 = torch.nn.Linear(93,64)
        self.linear2 = torch.nn.Linear(64,32)
        self.linear3 = torch.nn.Linear(32,16)
        self.linear4 = torch.nn.Linear(16,9)
        self.dropout = torch.nn.Dropout(0.3)

    def forward(self, x):
        x = torch.relu(self.linear1(x))
        x = self.dropout(x)
        x = torch.relu(self.linear2(x))
        x = self.dropout(x)
        x = torch.relu(self.linear3(x))
        x = self.dropout(x)
        return self.linear4(x)

    def predict(self, x):
        with torch.no_grad():
            x = torch.relu(self.linear1(x))
            x = torch.relu(self.linear2(x))
            x = torch.relu(self.linear3(x))
            x = self.linear4(x)
            x = torch.softmax(x, dim=1)
            return x


model = Model()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

def train(epoch):
    running_loss = 0.0
    for batch_idx, data in enumerate(train_loader,0):
        inputs, target = data
        inputs = inputs.float()
        target = target.long()
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if batch_idx % 300 == 299:
            print('[%d,%5d] loss:%.3f' % (epoch + 1, batch_idx + 1, running_loss / 300))
            running_loss = 0.0

def predict_save():
    test_dataset = pd.read_csv('./test.csv')
    test_inputs = torch.tensor(np.array(test_dataset)[:,1:].astype(float)).float()

    out = model.predict(test_inputs).numpy()
    # 自定义新的标签
    labels = ['Class_1', 'Class_2', 'Class_3', 'Class_4', 'Class_5', 'Class_6', 'Class_7', 'Class_8', 'Class_9']
    out = pd.DataFrame(out, columns=labels)
    # 插入id行
    out.insert(0, 'id', test_dataset['id'])
    out.to_csv('./my_predict.csv', index=False)
    return out


if __name__ == '__main__':
    model.train()
    for epoch in range(100):
        train(epoch)

    model.eval()
    predict_save()

