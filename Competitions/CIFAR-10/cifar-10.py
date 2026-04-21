import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os


class Cifar10Dataset(Dataset):
    def __init__(self, img_dir, label_csv=None):
        if label_csv:
            self.labels = pd.read_csv(label_csv)
            label_map = {label: idx for idx, label in enumerate(sorted(self.labels['label'].unique()))}
            self.y_data = torch.tensor([label_map[l] for l in self.labels['label']])
            self.ids = self.labels['id'].values
        else:
            self.ids = sorted([int(f.split('.')[0]) for f in os.listdir(img_dir) if f.endswith('.png')])
            self.y_data = None

        self.img_dir = img_dir
        images = []
        for img_id in self.ids:
            img = Image.open(os.path.join(img_dir, f'{img_id}.png'))
            images.append(np.array(img))
        self.x_data = torch.tensor(np.array(images)).permute(0, 3, 1, 2).float() / 255.0

    def __getitem__(self, index):
        if self.y_data is not None:
            return self.x_data[index], self.y_data[index]
        return self.x_data[index]

    def __len__(self):
        return len(self.ids)

class InceptionA(nn.Module):
    def __init__(self,in_Channels):
        super(InceptionA, self).__init__()
        self.branch1x1 = nn.Conv2d(in_Channels, 16, kernel_size=1)
        self.branch5x5_1 = nn.Conv2d(in_Channels, 16, kernel_size=1)
        self.branch5x5_2 = nn.Conv2d(16, 24, kernel_size=5, padding=2)
        self.branch3x3_1 = nn.Conv2d(in_Channels, 16, kernel_size=1)
        self.branch3x3_2 = nn.Conv2d(16, 24, kernel_size=3, padding=1)
        self.branch3x3_3 = nn.Conv2d(24, 24, kernel_size=3, padding=1)

        self.branch_pool = nn.Conv2d(in_Channels, 24, kernel_size=1)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)

        branch3x3 = self.branch3x3_1(x)
        branch3x3 = self.branch3x3_2(branch3x3)
        branch3x3 = self.branch3x3_3(branch3x3)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch5x5, branch3x3, branch_pool]
        return torch.cat(outputs, dim=1)

class ResidualBlock(nn.Module):
    def __init__(self,in_Channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_Channels,in_Channels,kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_Channels,in_Channels,kernel_size=3, padding=1)

    def forward(self, x):
        y = F.relu(self.conv1(x))
        y = self.conv2(y)
        return F.relu(y + x)

batch_size = 64

train_dataset = Cifar10Dataset('/Users/liyuan/Desktop/cifar-10/train', '/Users/liyuan/Desktop/cifar-10/trainLabels.csv')
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3,16,kernel_size=5)
        self.conv2 = nn.Conv2d(88,16,kernel_size=5)

        self.inception1 = InceptionA(in_Channels=16)
        self.inception2 = InceptionA(in_Channels=16)

        self.residual = ResidualBlock(in_Channels=16)

        self.mp = nn.MaxPool2d(2)
        self.fc = nn.Linear(2200,10)
        self.relu = nn.ReLU()

    def forward(self, x):
        in_size = x.size(0)
        x = self.mp(F.relu(self.conv1(x)))
        x = self.residual(x)
        x = self.inception1(x)
        x = self.mp(F.relu(self.conv2(x)))
        x = self.residual(x)
        x = self.inception2(x)
        x = x.view(in_size, -1)
        x = self.fc(x)
        return x

    def predict(self, x):
        with torch.no_grad():
            batch_size = x.size(0)
            x = self.mp(F.relu(self.conv1(x)))
            x = self.residual(x)
            x = self.inception1(x)
            x = self.mp(F.relu(self.conv2(x)))
            x = self.residual(x)
            x = self.inception2(x)
            x = x.view(batch_size,-1)
            x = self.fc(x)
            x = torch.softmax(x, dim=1)
            return x

net = Net()

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

def train(epoch):
    running_loss = 0.0
    for batch_idx, data in enumerate(train_loader,0):
        inputs, labels = data
        optimizer.zero_grad()

        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if batch_idx % 300 == 299:
            print('[%d,%5d] loss:%.3f' % (epoch + 1, batch_idx + 1, running_loss / 300))
            running_loss = 0.0

def predict_save():
    test_dataset = Cifar10Dataset('/Users/liyuan/Desktop/cifar-10/test')
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    all_preds = []
    with torch.no_grad():
        for inputs in test_loader:
            outputs = net.predict(inputs)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.numpy())

    label_map = {label: idx for idx, label in enumerate(sorted(pd.read_csv('/Users/liyuan/Desktop/cifar-10/trainLabels.csv')['label'].unique()))}
    idx_to_label = {v: k for k, v in label_map.items()}
    labels = [idx_to_label[p] for p in all_preds]

    submission = pd.DataFrame({'id': test_dataset.ids, 'label': labels})
    submission.to_csv('./cifar-10-predict.csv', index=False)
    return submission



if __name__ == '__main__':
    for epoch in range(30):
        train(epoch)
    predict_save()


