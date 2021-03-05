import sys
import os
import torch
from torch.autograd import Variable
import torch.autograd as autograd
import torch.utils.data
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from sklearn.utils import shuffle

batch_size = int(64)
n_input = int(40)
n_steps = int(50)
n_hidden = int(64)
n_layers = int(1)

use_cuda = torch.cuda.is_available()

print("cuda:", use_cuda)

class LSTM(nn.Module):
    def __init__(self, n_input, n_hidden, n_layers, n_steps, K, dropout=0.5, tie_weights=False):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(n_input, n_hidden, n_layers, dropout = dropout, batch_first = True)
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_layers = n_layers
        self.n_steps = n_steps
        self.hidden = self.init_hidden()
        self.fc = nn.Linear(n_hidden, K)

    def init_hidden(self, batch_size=64):
        return (autograd.Variable(torch.zeros(self.n_layers, batch_size, self.n_hidden).cuda()),
                autograd.Variable(torch.zeros(self.n_layers, batch_size, self.n_hidden).cuda()))

    def forward(self, xrd):
        xrd = xrd.view(-1, self.n_steps, self.n_input)
        lstm_out, self.hidden = self.lstm(xrd, self.hidden)
        y = self.fc(self.hidden[1].view(-1, self.n_hidden))
        y = F.log_softmax(y)
        return y

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

model = LSTM(n_input, n_hidden, n_layers, n_steps, K)
model.cuda()

criterion = torch.nn.CrossEntropyLoss(size_average=False)
optimizer = torch.optim.Adam(model.parameters(), lr = 1e-3)


def train(epoch):
    model.train()
    correct = 0
    train_loss = 0.0
    true_pos = 0
    n_pos = 0
    for batch_idx, (data, target) in enumerate(train_loader):
#        data = data.view(-1, n_steps, n_input)
        data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        model.hidden = model.init_hidden(data.size()[0])
        output = model(data)

        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        n_zeros = 0
        n_ones = 0
        for x in target.data:
            if x == 0:
                n_zeros += 1
            else:
                n_ones += 1
        
        if batch_idx % 50 == 0:
            print('Batch_idx: {}'.format(batch_idx))
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, batch_idx * len(data), len(train_loader.dataset), 100. * batch_idx / len(train_loader), loss.item()))

        train_loss += loss.item()
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

        for x1, x2 in zip(pred, target.data):
            if x2 == 0:
                n_pos += 1
                if x1[0] == 0:
                    true_pos += 1

    train_loss /= len(train_loader.dataset)
    print('Train set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%), True_pos: {:.6f}\n'.format(train_loss, correct, len(train_loader.dataset),100. * correct / len(train_loader.dataset), float(true_pos)/n_pos))


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
        model.hidden = model.init_hidden(data.size()[0])
        output = model(data)
        test_loss += criterion(output, target).item()
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

        for x1, x2 in zip(pred, target.data):
            if x1[0] == 0:
                x1_zeros += 1
            else:
                x1_ones += 1
            tot_pred.append(x1[0])
            if x2 == 0:
                n_pos += 1
                if x1[0] == 0:
                    true_pos += 1

    print("****************")
    print("0: %d, 1: %d" % (x1_zeros, x1_ones))
    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%), True_pos: {:.6f}'.format(test_loss, correct, len(test_loader.dataset),100. * correct / len(test_loader.dataset), float(true_pos)/n_pos))
    print("****************\n")

n_epoch = 50
for epoch in range(n_epoch):
    print("\n--------------------")
    print("start epoch:", epoch)
    print("--------------------\n")
    train(epoch)
    test()

if not os.path.exists("./model"):
    os.mkdir("./model")
torch.save(model.state_dict(), "model/lstm_%d_epoch_%d_1e-3_sum.torchmodel" % (K, n_epoch))
