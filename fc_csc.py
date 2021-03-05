import sys
import os
import torch
from torch.autograd import Variable
import torch.utils.data
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from sklearn.utils import shuffle
from pymatgen.ext.matproj import MPRester

len_feature = 2000

class CrystalNet(nn.Module):
    def __init__(self, n_input, K):
        super(CrystalNet, self).__init__()
        self.fc1 = torch.nn.Linear(n_input, 1000)
        self.fc2 = torch.nn.Linear(1000, 600)
        self.fc3 = torch.nn.Linear(600, 300)
        self.fc4 = torch.nn.Linear(300, K)

    def forward(self, x):
        x = x.view(-1, len_feature)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        y = self.fc4(x)
        return F.log_softmax(y)

def analyze(K):
    global features, labels
    cnts = np.zeros(K, int)
    for label in labels:
        cnts[label] += 1
    print("labels:", cnts)

print("Loading Data ......")
dataset = np.load('data/crystal_structures.npz')
features = np.array(dataset['xrd'], np.float32)
property = sys.argv[1]
if property == "crystal_system":
    labels = np.array(dataset['crystal_system'], int)
elif property == "space_group":
    labels = np.array(dataset['space_group'], int)-1 # 0-index
else:
    mps = np.array(dataset['mp'])
    m = MPRester("wdRVIWP0Sdc4Xp4pxQt")
    labels = []
    for mp in mps:
        val = m.query(criteria={"task_id": mp}, properties=[property])
        labels.append(val)
    labels = np.array(labels)

print(features.shape)
print(labels.shape, min(labels), max(labels), len(set(labels)))

K = max(labels)+1

features = features / np.amax(features, axis=1)[:, None]
analyze(K)

features, labels = shuffle(features, labels, random_state = 0)

print("Generating data_loader ......")

n_samples = len(features)
cutoff = int(n_samples*9.0/10)
X_train = features[:cutoff, :]
Y_train = labels[:cutoff]

X_test = features[cutoff:, :]
Y_test = labels[cutoff:]

train_tensor = torch.utils.data.TensorDataset(torch.from_numpy(X_train), torch.from_numpy(Y_train))
train_loader = torch.utils.data.DataLoader(train_tensor, batch_size=64, shuffle=True)

test_tensor = torch.utils.data.TensorDataset(torch.from_numpy(X_test), torch.from_numpy(Y_test))
test_loader = torch.utils.data.DataLoader(test_tensor, batch_size=64, shuffle=True)

model = CrystalNet(len_feature, K)
model.cuda()

criterion = torch.nn.CrossEntropyLoss(reduction='sum')
optimizer = torch.optim.Adam(model.parameters(), lr = 1e-3)

def train(epoch):
    model.train()
    correct = 0
    train_loss = 0.0
    true_pos = 0
    n_pos = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)

        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

        if batch_idx % 50 == 0:
            print('Batch_idx: {}'.format(batch_idx))
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, batch_idx * len(data), len(train_loader.dataset), 100. * batch_idx / len(train_loader), loss.item()))

        train_loss += F.nll_loss(output, target, reduction='sum').item()
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    train_loss /= len(train_loader.dataset)
    print('Train set: Average loss: {:.4f}, Accuracy: {}/{} ({:.3f}%)\n'.format(train_loss, correct, len(train_loader.dataset),100. * correct / len(train_loader.dataset)))

def test():
    model.eval()
    test_loss = 0
    correct = 0
    true_pos = 0
    n_pos = 0
    tot_pred = []
    x1_zeros = 0
    x1_ones = 0
    for data, target in test_loader:
        data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        output = model(data)
        test_loss += F.nll_loss(output, target, reduction='sum').item()
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    print("****************")
    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.3f}%)'.format(test_loss, correct, len(test_loader.dataset),100. * correct / len(test_loader.dataset)))
    print("****************\n")

n_epoch = 50
for epoch in range(n_epoch):
    print("\n--------------------")
    print("start epoch:", epoch)
    print("--------------------\n")
    train(epoch)
    if epoch % 1 == 0:
        test()

if not os.path.exists("./model"):
    os.mkdir("./model")
torch.save(model.state_dict(), "model/fc_%d_epoch_%d.torchmodel" % (K, n_epoch))
