import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import datasets

np.random.seed(0)
torch.manual_seed(0)

data_task1 = np.load('data_task1.npy')
label_task1 = np.load('label_task1.npy')
X = data_task1
y = label_task1
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
X_train = torch.FloatTensor(X_train)
X_test = torch.FloatTensor(X_test)
y_train = torch.LongTensor(y_train)
y_test = torch.LongTensor(y_test)

class ANN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(in_features=20, out_features=16)
        self.fc2 = nn.Linear(in_features=16, out_features=8)
        self.output = nn.Linear(in_features=8, out_features=4)
    def forward(self, x):
        x = F.elu(self.fc1(x))
        x = F.elu(self.fc2(x))
        x = self.output(x)
        return x
   
model = ANN()
model
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
epochs = 100
loss_arr = []
train_loss = []
train_accuracy = []

for i in range(epochs): 
    y_hat = model(X_train)
    loss = criterion(y_hat, y_train)
    
    y_hat_class = np.argmax(y_hat.data.numpy(),axis=1)
    accuracy = np.sum(y_train.data.numpy()==y_hat_class) / len(y_train) 
    train_accuracy.append(accuracy)
    train_loss.append(loss.item())
    if i % 10 == 0:
        print(f'Epoch: {i} Loss: {loss}')
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    
# plot the training loss and training accuracy
fig, ax = plt.subplots(2, 1, figsize=(12,8))
ax[0].plot(train_loss)
ax[0].set_ylabel('Loss')
ax[0].set_title('Training Loss')

ax[1].plot(train_accuracy)
ax[1].set_ylabel('Classification Accuracy')
ax[1].set_title('Training Accuracy')

plt.tight_layout()
plt.show()

y_hat_test = model(X_test)

y_hat_class_test = np.argmax(y_hat_test.data.numpy(),axis=1)
test_accuracy = np.sum(y_test.data.numpy()==y_hat_class_test) / len(y_test)

print("Test Accuracy {:.2f}".format(test_accuracy))

