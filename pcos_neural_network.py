#!/usr/bin/env python
# coding: utf-8

# # import libraries



import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score,classification_report
import pickle
import torch
from torch import nn
from torch.utils.data import Dataset,DataLoader,TensorDataset
import torch.optim as optim


# # load data

data_clean = pd.read_csv(open("data_tidy.csv",'rb'))

X = data_clean.iloc[:,2:]
y = data_clean.iloc[:,1]
X_train,X_test, y_train, y_test = train_test_split(X,y, test_size=0.3,random_state = 36) 
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# ## Neural Network

# In[16]:


X_train_tensor = torch.FloatTensor(X_train)
y_train_tensor = torch.FloatTensor(y_train).view(-1,1)
X_test_tensor = torch.FloatTensor(X_test)
y_test_tensor = torch.FloatTensor(np.array(y_test)).view(-1,1)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)


# In[29]:


device = 'cpu'
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork,self).__init__()
        self.fc1 = nn.Linear(38, 64)  # 输入层到隐藏层
        self.fc2 = nn.Linear(64, 64)   # 隐藏层到隐藏层
        self.fc3 = nn.Linear(64, 1)    # 隐藏层到输出层
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x

model = NeuralNetwork().to(device)


# In[30]:


model = NeuralNetwork()
criterion = nn.BCELoss()  # 二元交叉熵损失
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 10
for epoch in range(num_epochs):
    total_loss = 0
    for inputs, labels in train_loader:
        optimizer.zero_grad()  # 清零梯度
        outputs = model(inputs)  # 前向传播
        loss = criterion(outputs, labels)  # 计算损失
        loss.backward()  # 反向传播
        optimizer.step()  # 更新参数
        total_loss += loss.item()

    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss:.4f}')

# 评估模型
with torch.no_grad():
    model.eval()  # 切换到评估模式
    test_outputs = model(X_test_tensor)
    predicted = (test_outputs > 0.5).float()  # 将输出转换为0或1
    accuracy = (predicted.eq(y_test_tensor).sum().item()) / y_test_tensor.size(0)
    print(f'Test accuracy: {accuracy:.4f}')





