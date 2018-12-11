import torch
import torch.nn as nn
import torch.utils.data as data_utils
import torchvision.datasets as dsets
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.utils.data.sampler import SubsetRandomSampler
from torch.autograd import Variable
import time
import os.path
import numpy as np
import pickle
from tqdm import *


num_epochs = 500
batch_size = 32
learning_rate = 0.001
print_every = 1
best_accuracy = torch.FloatTensor([0])
start_epoch = 0
num_input_channel = 3
num_output_channel = 2

resume_weights = "sample_data/checkpointSEG.pth.tar"

cuda = torch.cuda.is_available()

torch.manual_seed(1)

if cuda:
    torch.cuda.manual_seed(1)

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    torchvision.transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor()
])

imsize = 64

print("Loading the dataset")
train_set = torchvision.datasets.ImageFolder(root="Dataset_1", transform=transform)
train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=batch_size, shuffle=False)


test_set = torchvision.datasets.ImageFolder(root="Dataset_1", transform=transform)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)

print("Dataset is loaded")

print("Saving the dataset...")
pickle.dump(train_loader, open("sample_data/train_loader.txt", 'wb'))
pickle.dump(test_loader, open("sample_data/test_loader.txt", 'wb'))

print(len(train_loader))
print(len(test_loader))
print("Dataset is saved")


def train(epoch):
    model.train()

    # update learning rate
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # define a weighted loss (0 weight for 0 label)
    weights_list = [0]+[1 for i in range(17)]
    weights = np.asarray(weights_list)
    weigthtorch = torch.Tensor(weights_list)

    if cuda:
        loss = nn.CrossEntropyLoss(weight=weigthtorch).cuda()
    else:
        loss = nn.CrossEntropyLoss(weight=weigthtorch)

    total_loss = 0

    for batch_idx, batch_files in enumerate(tqdm(train_loader)):

        # containers
        batch = np.zeros((batch_size, num_input_channel, imsize, imsize), dtype=float)
        batch_labels = np.zeros((batch_size,imsize, imsize), dtype=int)

        # fill the batch
        # ...

        batch_th = Variable(torch.Tensor(batch_idx))
        target_th = Variable(torch.LongTensor(batch_files))

        if cuda:
            batch_th = batch_th.cuda()
            target_th = target_th.cuda()

        # initilize gradients
        optimizer.zero_grad()

        # predictions
        output = model(batch_th)

        # Loss
        output = output.view(output.size(0),output.size(1), -1)
        output = torch.transpose(output,1,2).contiguous()
        output = output.view(-1,output.size(2))
        target = target_th.view(-1)

        l_ = loss(output.cuda(), target)
        total_loss += l_.cpu().data.numpy()
        l_.cuda()
        l_.backward()
        optimizer.step()

    return total_loss/len(files)


def eval(model, test_loader):
    model.eval()

    acc = 0
    total = 0
    for i, (data, labels) in enumerate(test_loader):
        data, labels = Variable(data), Variable(labels)
        if cuda:
            data, labels = data.cuda(), labels.cuda()

        data = data.squeeze(0)
        labels = labels.squeeze(0)

        outputs = model(data)
        if cuda:
            outputs.cpu()

        total += labels.size(0)
        prediction = outputs.data.max(1)[1]
        correct = prediction.eq(labels.data).sum()
        acc += correct
    return acc / total


def save_checkpoint(state, is_best, filename="sample_data/checkpointSEG.pth.tar"):
    if is_best:
        print("=> Saving a new best")
        torch.save(state, filename)
    else:
        print("=> Validation Accuracy did not improve")


class SegNet(nn.Module):
    def __init__(self, input_nbr, label_nbr):
        super(SegNet, self).__init__()

        batchNorm_momentum = 0.1

        self.conv11 = nn.Conv2d(input_nbr, 64, kernel_size=3, padding=1)
        self.bn11 = nn.BatchNorm2d(64, momentum=batchNorm_momentum)
        self.conv12 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn12 = nn.BatchNorm2d(64, momentum=batchNorm_momentum)

        self.conv21 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn21 = nn.BatchNorm2d(128, momentum=batchNorm_momentum)
        self.conv22 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn22 = nn.BatchNorm2d(128, momentum=batchNorm_momentum)

        self.conv31 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn31 = nn.BatchNorm2d(256, momentum=batchNorm_momentum)
        self.conv32 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn32 = nn.BatchNorm2d(256, momentum=batchNorm_momentum)
        self.conv33 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn33 = nn.BatchNorm2d(256, momentum=batchNorm_momentum)

        self.conv41 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn41 = nn.BatchNorm2d(512, momentum=batchNorm_momentum)
        self.conv42 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn42 = nn.BatchNorm2d(512, momentum=batchNorm_momentum)
        self.conv43 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn43 = nn.BatchNorm2d(512, momentum=batchNorm_momentum)

        self.conv51 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn51 = nn.BatchNorm2d(512, momentum=batchNorm_momentum)
        self.conv52 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn52 = nn.BatchNorm2d(512, momentum=batchNorm_momentum)
        self.conv53 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn53 = nn.BatchNorm2d(512, momentum=batchNorm_momentum)

        self.conv53d = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn53d = nn.BatchNorm2d(512, momentum=batchNorm_momentum)
        self.conv52d = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn52d = nn.BatchNorm2d(512, momentum=batchNorm_momentum)
        self.conv51d = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn51d = nn.BatchNorm2d(512, momentum=batchNorm_momentum)

        self.conv43d = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn43d = nn.BatchNorm2d(512, momentum=batchNorm_momentum)
        self.conv42d = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn42d = nn.BatchNorm2d(512, momentum=batchNorm_momentum)
        self.conv41d = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.bn41d = nn.BatchNorm2d(256, momentum=batchNorm_momentum)

        self.conv33d = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn33d = nn.BatchNorm2d(256, momentum=batchNorm_momentum)
        self.conv32d = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn32d = nn.BatchNorm2d(256, momentum=batchNorm_momentum)
        self.conv31d = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.bn31d = nn.BatchNorm2d(128, momentum=batchNorm_momentum)

        self.conv22d = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn22d = nn.BatchNorm2d(128, momentum=batchNorm_momentum)
        self.conv21d = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.bn21d = nn.BatchNorm2d(64, momentum=batchNorm_momentum)

        self.conv12d = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn12d = nn.BatchNorm2d(64, momentum=batchNorm_momentum)
        self.conv11d = nn.Conv2d(64, label_nbr, kernel_size=3, padding=1)

    def forward(self, x):
        # Stage 1
        x11 = F.relu(self.bn11(self.conv11(x)))
        x12 = F.relu(self.bn12(self.conv12(x11)))
        x1p, id1 = F.max_pool2d(x12, kernel_size=2, stride=2, return_indices=True)

        # Stage 2
        x21 = F.relu(self.bn21(self.conv21(x1p)))
        x22 = F.relu(self.bn22(self.conv22(x21)))
        x2p, id2 = F.max_pool2d(x22, kernel_size=2, stride=2, return_indices=True)

        # Stage 3
        x31 = F.relu(self.bn31(self.conv31(x2p)))
        x32 = F.relu(self.bn32(self.conv32(x31)))
        x33 = F.relu(self.bn33(self.conv33(x32)))
        x3p, id3 = F.max_pool2d(x33, kernel_size=2, stride=2, return_indices=True)

        # Stage 4
        x41 = F.relu(self.bn41(self.conv41(x3p)))
        x42 = F.relu(self.bn42(self.conv42(x41)))
        x43 = F.relu(self.bn43(self.conv43(x42)))
        x4p, id4 = F.max_pool2d(x43, kernel_size=2, stride=2, return_indices=True)

        # Stage 5
        x51 = F.relu(self.bn51(self.conv51(x4p)))
        x52 = F.relu(self.bn52(self.conv52(x51)))
        x53 = F.relu(self.bn53(self.conv53(x52)))
        x5p, id5 = F.max_pool2d(x53, kernel_size=2, stride=2, return_indices=True)

        # Stage 5d
        x5d = F.max_unpool2d(x5p, id5, kernel_size=2, stride=2)
        x53d = F.relu(self.bn53d(self.conv53d(x5d)))
        x52d = F.relu(self.bn52d(self.conv52d(x53d)))
        x51d = F.relu(self.bn51d(self.conv51d(x52d)))

        # Stage 4d
        x4d = F.max_unpool2d(x51d, id4, kernel_size=2, stride=2)
        x43d = F.relu(self.bn43d(self.conv43d(x4d)))
        x42d = F.relu(self.bn42d(self.conv42d(x43d)))
        x41d = F.relu(self.bn41d(self.conv41d(x42d)))

        # Stage 3d
        x3d = F.max_unpool2d(x41d, id3, kernel_size=2, stride=2)
        x33d = F.relu(self.bn33d(self.conv33d(x3d)))
        x32d = F.relu(self.bn32d(self.conv32d(x33d)))
        x31d = F.relu(self.bn31d(self.conv31d(x32d)))

        # Stage 2d
        x2d = F.max_unpool2d(x31d, id2, kernel_size=2, stride=2)
        x22d = F.relu(self.bn22d(self.conv22d(x2d)))
        x21d = F.relu(self.bn21d(self.conv21d(x22d)))

        # Stage 1d
        x1d = F.max_unpool2d(x21d, id1, kernel_size=2, stride=2)
        x12d = F.relu(self.bn12d(self.conv12d(x1d)))
        x11d = self.conv11d(x12d)

        return x11d


model = SegNet(num_input_channel, num_output_channel)

if cuda:
    model.cuda()

criterion = nn.CrossEntropyLoss()
if cuda:
    criterion.cuda()

total_step = len(train_loader)

if os.path.isfile(resume_weights):
    print("=> loading checkpoint '{}' ...".format(resume_weights))
    if cuda:
        checkpoint = torch.load(resume_weights)
    else:
        checkpoint = torch.load(resume_weights, map_location=lambda storage, loc: storage)
    start_epoch = checkpoint['epoch']
    best_accuracy = checkpoint['best_accuracy']
    model.load_state_dict(checkpoint['state_dict'])
    print("=> loaded checkpoint '{}' (trained for {} epochs)".format(resume_weights, checkpoint['epoch']))

for epoch in range(num_epochs):
    print(learning_rate)

    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.5)

    if learning_rate >= 0.0003:
        learning_rate = learning_rate * 0.1

    train(model, optimizer, train_loader, criterion)
    acc = eval(model, test_loader)
    print('=> Validation set: Accuracy: {:.2f}%'.format(acc * 100))
    acc = torch.FloatTensor([acc])

    is_best = bool(acc.numpy() > best_accuracy.numpy())

    best_accuracy = torch.FloatTensor(max(acc.numpy(), best_accuracy.numpy()))

    save_checkpoint({
        'epoch': start_epoch + epoch + 1,
        'state_dict': model.state_dict(),
        'best_accuracy': best_accuracy
    }, is_best)

test_acc = eval(model, test_loader)
print('=> Test set: Accuracy: {:.2f}%'.format(test_acc * 100))

