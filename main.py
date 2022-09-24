import torch
import torch.nn as nn
from torch_geometric.datasets import Planetoid
import networkx as nx
import matplotlib.pyplot as plt
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv
import seaborn as sns

data_name = 'cora'
dataset = Planetoid(root='/data', name=data_name, transform=T.NormalizeFeatures())
data = dataset[0]
print("attibute of data {}".format(data))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class GCN(nn.Module):
    def __init__(self, in_feature, hidden_feature, out_feature, dropout):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_feature, hidden_feature)
        self.bns = torch.nn.BatchNorm1d(hidden_feature)
        self.conv2 = GCNConv(hidden_feature, hidden_feature)
        self.lkRelu = nn.LeakyReLU(0.2)
        self.linear = nn.Linear(hidden_feature, out_feature)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = self.lkRelu(self.bns(self.conv1(x, edge_index)))
        x = self.lkRelu(self.bns(self.conv2(x, edge_index)))
        x = self.linear(x)
        return x


model = GCN(dataset.num_features, dataset.num_features // 2, dataset.num_classes, 0.5)
critetion = torch.nn.CrossEntropyLoss()
optim = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0.9)


def Acc(pred, label):
    pred = pred.argmax(dim=1)
    return torch.sum(pred == label) / len(pred) * 100


def train(model, optimizer, critetion, data):
    model.train()
    optimizer.zero_grad()
    model = model.to(device)
    x = data.x.to(device)
    y = data.y.to(device)
    edge_index = data.edge_index.to(device)
    out = model(x, edge_index)
    out = out[data.train_mask]
    y = y[data.train_mask]
    loss = critetion(out, y)
    loss.backward()
    optimizer.step()
    return loss, Acc(out, y)


def eval(model, data):
    model = model.to("cpu")
    model.eval()
    x, y = data.x, data.y
    out = model(x, data.edge_index)
    out = out[data.val_mask]
    y = y[data.val_mask]
    return Acc(out, y)


loss_train = []
loss_eval = []
for i in range(100):
    l, acc = train(model, optimizer=optim, critetion=critetion, data=data)
    print("Epoch {} has loss {} train acc {:2f}% val acc {:2f}%".format(i + 1, l, acc, eval(model, data)))
    loss_train.append(l)
    loss_eval.append(eval(model, data))

