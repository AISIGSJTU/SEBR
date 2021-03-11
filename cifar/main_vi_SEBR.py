#!/usr/bin/env python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.optim import Adam, SGD
import torchvision
import torchvision.transforms as transforms

import math
import os
import argparse

from utils.loss import elbo

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
parser.add_argument('--sigma_0', required=True, type=float, help='Gaussian prior')
parser.add_argument('--init_s', required=True, type=float, help='Initial log(std) of posterior')
parser.add_argument('--data', required=True, type=str, help='dataset name')
parser.add_argument('--model', required=True, type=str, help='model name')
parser.add_argument('--root', required=True, type=str, help='path to dataset')
parser.add_argument('--model_out', required=True, type=str, help='output path')
parser.add_argument('--resume', action='store_true', help='resume')
opt = parser.parse_args()


def expectation_spectral_norm_upper_bound_calculation(W_mu, W_p=None, SIMU_TIMES=10, ITERATION_TIMES=10):
    u = torch.rand(W_mu.shape[0]).cuda()
    v = torch.rand(W_mu.shape[1]).cuda()
    for _ in range(ITERATION_TIMES):
        v = torch.nn.functional.normalize(torch.mv(W_mu.t(), u), dim=0, eps=1e-12)
        u = torch.nn.functional.normalize(torch.mv(W_mu, v), dim=0, eps=1e-12)
    sigma = torch.dot(u, torch.mv(W_mu, v))
    if W_p is None:
        return sigma

    std_w = 1e-6 + F.softplus(W_p, beta=1, threshold=20)

    res = torch.max(torch.norm(std_w, dim=1)) + torch.max(torch.norm(std_w, dim=0))

    tmp = 0
    for _ in range(SIMU_TIMES):
        eps_W = W_mu.data.new(W_mu.size()).normal_()
        tmp += torch.max(1 * eps_W * std_w)
    tmp /= SIMU_TIMES
    return res + tmp + sigma


opt.init_s = math.log(opt.init_s)  # init_s is log(std)
# Data
print('==> Preparing data..')
if opt.data == 'cifar10':
    nclass = 10
    img_width = 32
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])
    trainset = torchvision.datasets.CIFAR10(root=opt.root, train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
    testset = torchvision.datasets.CIFAR10(root=opt.root, train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)
elif opt.data == 'cifar100':
    nclass = 100
    img_width = 32
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])
    trainset = torchvision.datasets.CIFAR100(root=opt.root, train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
    testset = torchvision.datasets.CIFAR100(root=opt.root, train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)
else:
    raise NotImplementedError('Invalid dataset')

print(len(trainset), len(testset))

# Model
if opt.model == 'vgg':
    from models.vgg_vi import VGG

    net = nn.DataParallel(VGG(opt.sigma_0, len(trainset), opt.init_s, 'VGG16', nclass, img_width=img_width).cuda())
else:
    raise NotImplementedError('Invalid model')

if opt.resume:
    print(f'==> Resuming from {opt.model_out}')
    net.load_state_dict(torch.load(opt.model_out))

cudnn.benchmark = True


def get_beta(epoch_idx, N):
    return 1.0 / N / 100


# Training
def train(epoch):
    print('Epoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.cuda(), targets.cuda()
        optimizer.zero_grad()
        outputs, kl = net(inputs)
        loss = elbo(outputs, targets, kl, get_beta(epoch, len(trainset)))
        layer = net.module.classifier
        loss += 0.1 * expectation_spectral_norm_upper_bound_calculation(layer.mu_weight, layer.sigma_weight)
        loss.backward()
        optimizer.step()
        pred = torch.max(outputs, dim=1)[1]
        correct += torch.sum(pred.eq(targets)).item()
        total += targets.numel()
    print(f'[TRAIN] Acc: {100. * correct / total:.3f}')


def test(epoch):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs, _ = net(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        print(f'[TEST] Acc: {100. * correct / total:.3f}')
    # Save checkpoint.
    torch.save(net.state_dict(), opt.model_out)


if opt.data == 'cifar10' or opt.data == 'cifar100':
    epochs = [80, 60, 40, 20]
elif opt.data == 'imagenet-sub':
    epochs = [30, 20, 20, 10]
elif opt.data == 'fashion':
    epochs = [40, 30, 20]
elif opt.data == 'stl10':
    epochs = [60, 40, 20]
count = 0

for epoch in epochs:
    optimizer = Adam(net.parameters(), lr=opt.lr)
    for _ in range(epoch):
        train(count)
        test(count)
        count += 1
    opt.lr /= 10
