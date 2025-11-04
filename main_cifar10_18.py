'''Train CIFAR10 with PyTorch.'''
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from utils import *
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import os
import argparse
import numpy as np
import sys
import logging
from loralib.utils import *
# from gl_model.nl_lora import lora_nl101, lora_nl18
from gm_model.nlsin_ft import monomial_ft_nlsin101, monomial_ft_nlsin18
from groupnl.nlsin_resnet import nlsin_resnet101, SinBottleneck, nlsin_resnet18, SinBasicBlock

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

CORRUPTIONS = load_txt('/home/u2318483074/nlnl/corruptions.txt')
MEAN = [0.49139968, 0.48215841, 0.44653091]
STD = [0.24703223, 0.24348513, 0.26158784]

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.025, type=float, help='learning rate')
parser.add_argument('--wd', default=1e-3, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
parser.add_argument('--epochs', default=300, type=int, help='number of training epochs')
parser.add_argument('--model', default='groupnl_18_monoft', type=str,
                    help='which model')
parser.add_argument('--exp_factor', default=[2, 2, 2, 2], nargs='+', type=int, help='reduction factor')
parser.add_argument('--exp_range', nargs="*", default=(1, 3), type=float,
                    help='minimum and maximal bound of exp')
parser.add_argument('--num_terms_mono', default=1, type=int, help='number of terms')
parser.add_argument('--num_classes', default=10, type=int, help='number of classes')
parser.add_argument('--filter_groups', default=1, type=int, help='number of groups')
parser.add_argument('--channel_groups', default=1, type=int, help='number of groups')
parser.add_argument('--r', default=64, type=int, help='rank of matrices A and B in lora')
parser.add_argument('--lora_alpha', default=128, type=int, help='lora alpha value')
parser.add_argument('--alpha', default=[0.999, 0.999, 0.999, 0.999], nargs='+', type=float, help='reduction factor')
parser.add_argument('--alpha_learn', default=False, type=lambda x: x.lower() == 'true', help='whether to learn alpha')
parser.add_argument('--alpha_range', nargs="*", default=(0, 128), type=float,
                    help='minimum and maximal bound of eps')
parser.add_argument('--rbf', default='gaussian', choices=['mono', 'gaussian', 'legendre', 'sin',
'mono_gaussian_legendre_sin', 'mono_gaussian_sin', 'mono_legendre_sin', 'gaussian_legendre_sin', 'mono_gaussian', 'mono_legendre', 'mono_sin',
'gaussian_legendre', 'gaussian_sin', 'legendre_sin'], type=str, help='type of nlf')
parser.add_argument('--onebyone', default=False, type=lambda x: x.lower() == 'true', help='whether to use 1×1 conv')
parser.add_argument('--data', default='tmp', type=str, help='directory of dataset')
parser.add_argument('--data_root', type=str, default='/home/u2318483074/nlnl/datasets', help='root path to cifar10-c directory')
parser.add_argument( '--corruptions', type=str, nargs='*', default=CORRUPTIONS, help='testing corruption types')
parser.add_argument('--output_save_dir', default='tmp', type=str, help='directory for saving output')
parser.add_argument('--model_save_dir', default='tmp', type=str, help='directory for saving model')
parser.add_argument('--picture_save_dir', default='tmp', type=str, help='directory for saving loss and acc picture')
parser.add_argument('--cifar10_c_save_dir', default='tmp', type=str, help='directory for saving cifar10-c picture')
parser.add_argument('--batch_size', default=128, type=int, help='number of samples in one iteration')
parser.add_argument("--run_rank", default=1, type=int, help='rank of run')
parser.add_argument('--seed', default=2024, type=int, help='random seed')

parser.add_argument('--reduction_ratio', default=2, type=int, help='groupnl')
parser.add_argument('--period_range', nargs="*", default=(1, 2), type=float, help='minimum and maximal bound of period')
parser.add_argument('--shift_range', nargs="*", default=(1, 5), type=float, help='minimum and maximal bound of shift')
parser.add_argument('--num_terms', default=4, type=int, help='groupnl')
parser.add_argument('--shuffle', default=False, type=lambda x: x.lower() == 'true', help='groupnl')
parser.add_argument('--learn', default=False, type=lambda x: x.lower() == 'true', help='groupnl')

args = parser.parse_args()

seed_everything(args.seed)
if args.model in ['groupnl_18_monoft']:
    alpha_part = "alpha_range{}".format(args.alpha_range) if args.alpha_learn else "alpha{}".format(args.alpha)
    args.output_save_dir = '/home/u2318483074/nlnl/exps2/cifar10/log/{}_r{}_fgroups{}_cgroups{}_{}_exp{}{}_monoterms{}_bs{}_lr{}_wd{}_epochs{}_cos_run{}.log'.format(
        args.model, str(args.exp_factor), str(args.filter_groups), str(args.channel_groups), alpha_part,
        str(args.exp_range[0]), str(args.exp_range[1]), str(args.num_terms_mono), str(args.batch_size), str(args.lr), str(args.wd),
        str(args.epochs), str(args.run_rank))
    args.model_save_dir = '/home/u2318483074/nlnl/exps2/cifar10/pth/{}_r{}_fgroups{}_cgroups{}_{}_exp{}{}_monoterms{}_bs{}_lr{}_wd{}_epochs{}_cos_run{}.pth'.format(
        args.model, str(args.exp_factor), str(args.filter_groups), str(args.channel_groups), alpha_part,
        str(args.exp_range[0]), str(args.exp_range[1]), str(args.num_terms_mono), str(args.batch_size), str(args.lr), str(args.wd),
        str(args.epochs), str(args.run_rank))
    args.picture_save_dir = '/home/u2318483074/nlnl/exps2/cifar10/plots/{}_r{}_fgroups{}_cgroups{}_{}_exp{}{}_monoterms{}_bs{}_lr{}_wd{}_epochs{}_cos_run{}.png'.format(
        args.model, str(args.exp_factor), str(args.filter_groups), str(args.channel_groups), alpha_part,
        str(args.exp_range[0]), str(args.exp_range[1]), str(args.num_terms_mono), str(args.batch_size), str(args.lr), str(args.wd),
        str(args.epochs), str(args.run_rank))
    args.cifar10_c_save_dir = '/home/u2318483074/nlnl/exps2/cifar10c/groupnl_18_monoft'

elif args.model in ['groupnl_18']:
    args.output_save_dir = '/home/u2318483074/nlnl/exps/cifar10/groupnl_18/log/period{}{}_shift{}{}_numterm{}_redu{}.log'.format(
        str(args.period_range[0]),str(args.period_range[1]),str(args.shift_range[0]),str(args.shift_range[1]),str(args.num_terms),str(args.reduction_ratio))
    args.model_save_dir = '/home/u2318483074/nlnl/exps/cifar10/groupnl_18/pth/period{}{}_shift{}{}_numterm{}_redu{}.pth'.format(
        str(args.period_range[0]),str(args.period_range[1]),str(args.shift_range[0]),str(args.shift_range[1]),str(args.num_terms),str(args.reduction_ratio))
    args.picture_save_dir = '/home/u2318483074/nlnl/exps/cifar10/groupnl_18/plots/period{}{}_shift{}{}_numterm{}_redu{}.png'.format(
        str(args.period_range[0]),str(args.period_range[1]),str(args.shift_range[0]),str(args.shift_range[1]),str(args.num_terms),str(args.reduction_ratio))

elif args.model in ['groupnl_18_FT']:
    args.output_save_dir = '/home/u2318483074/nlnl/exps/cifar10/groupnl_18_FT/log/rank{}_alpha{}.log'.format(
        str(args.r), str(args.lora_alpha))
    args.model_save_dir = '/home/u2318483074/nlnl/exps/cifar10/groupnl_18_FT/pth/rank{}_alpha{}.pth'.format(
        str(args.r), str(args.lora_alpha))
    args.picture_save_dir = '/home/u2318483074/nlnl/exps/cifar10/groupnl_18_FT/plots/rank{}_alpha{}.png'.format(
        str(args.r), str(args.lora_alpha))

elif args.model in ['groupnl_18_lora']:
    args.output_save_dir = '/home/u2318483074/nlnl/exps/cifar10/groupnl_18_lora/log/rank{}_loraalpha{}.log'.format(str(args.r), str(args.lora_alpha))
    args.model_save_dir = '/home/u2318483074/nlnl/exps/cifar10/groupnl_18_lora/pth/rank{}_loraalpha{}.pth'.format(str(args.r), str(args.lora_alpha))
    args.picture_save_dir = '/home/u2318483074/nlnl/exps/cifar10/groupnl_18_lora/plots/rank{}_loraalpha{}.png'.format(str(args.r), str(args.lora_alpha))

args.data = '/home/u2318483074/nlnl/datasets/cifar10'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
device = 'cuda' if torch.cuda.is_available() else 'cpu'


os.makedirs(os.path.dirname(args.output_save_dir), exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(args.output_save_dir),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# Data
logger.info('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.Resize(32),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(
    root=args.data, train=True, download=False, transform=transform_train)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(
    root=args.data, train=False, download=False, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=args.batch_size, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

# Model
logger.info('==> Building model..')
if args.model == 'groupnl_18_monoft':
    checkpoint_path = '/home/u2318483074/nlnl/nl_checkpoint/nl18/checkpoint-303.pth.tar'
    checkpoint = torch.load(checkpoint_path, map_location='cuda')
    pretrained_model = nlsin_resnet18(reduction_ratio=args.reduction_ratio, period_range=args.period_range,
                                       shift_range=args.shift_range,
                                       num_terms=args.num_terms, learn=args.learn, shuffle=args.shuffle,
                                       num_classes=args.num_classes)
    model_dict = pretrained_model.state_dict()
    pretrained_dict = checkpoint['state_dict']
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and v.size() == model_dict[k].size()}
    model_dict.update(pretrained_dict)
    pretrained_model.load_state_dict(model_dict)
    pretrained_model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    pretrained_model.fc = nn.Linear(512 * SinBasicBlock.expansion, 10)
    net = monomial_ft_nlsin18(model=pretrained_model, reduction_ratio=args.reduction_ratio,
                               period_range=args.period_range, shift_range=args.shift_range,
                               num_terms=args.num_terms, shuffle=args.shuffle, learn=args.learn,
                               exp_range=args.exp_range, alpha_range=args.alpha_range,
                               exp_factor=args.exp_factor, alpha_learn=args.alpha_learn,
                               num_terms_mono=args.num_terms_mono, filter_groups=args.filter_groups,
                               channel_groups=args.channel_groups, mono_bias=False, onebyone=args.onebyone,
                               alpha=args.alpha, num_classes=args.num_classes).to(device)

elif args.model == 'groupnl_18':
    net = nlsin_resnet18(reduction_ratio=args.reduction_ratio, period_range=args.period_range,
                          shift_range=args.shift_range, num_terms=args.num_terms,
                          shuffle=args.shuffle, learn=args.learn, num_classes=args.num_classes).to(device)

elif args.model == 'groupnl_18_FT':
    checkpoint_path = '/home/u2318483074/nlnl/nl_checkpoint/nl18/checkpoint-303.pth.tar'
    checkpoint = torch.load(checkpoint_path, map_location='cuda')
    net = nlsin_resnet18(reduction_ratio=args.reduction_ratio, period_range=args.period_range, shift_range=args.shift_range,
                         num_terms=args.num_terms, learn=args.learn, shuffle=args.shuffle, num_classes=args.num_classes)
    model_dict = net.state_dict()
    pretrained_dict = checkpoint['state_dict']
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and v.size() == model_dict[k].size()}
    model_dict.update(pretrained_dict)
    net.load_state_dict(model_dict)
    net.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    net.fc = nn.Linear(512 * SinBasicBlock.expansion, 10)

elif args.model == 'groupnl_18_lora':
    checkpoint_path = '/home/u2318483074/nlnl/nl_checkpoint/nl18/checkpoint-303.pth.tar'
    checkpoint = torch.load(checkpoint_path, map_location='cuda')
    pretrained_model = nlsin_resnet18(reduction_ratio=args.reduction_ratio, period_range=args.period_range, shift_range=args.shift_range,
                   num_terms=args.num_terms, learn=args.learn, shuffle=args.shuffle, num_classes=args.num_classes)
    model_dict = pretrained_model.state_dict()
    pretrained_dict = checkpoint['state_dict']
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and v.size() == model_dict[k].size()}
    model_dict.update(pretrained_dict)
    pretrained_model.load_state_dict(model_dict)
    pretrained_model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    pretrained_model.fc = nn.Linear(512 * SinBasicBlock.expansion, 10)
    net = lora_nl18(model=pretrained_model, reduction_ratio=args.reduction_ratio, period_range=args.period_range, shift_range=args.shift_range,
                   num_terms=args.num_terms, shuffle=args.shuffle, learn=args.learn, num_classes=args.num_classes, r=args.r, lora_alpha=args.lora_alpha).to(device)

else:
    raise NotImplementedError

total_params = count_parameters(net)
logger.info("Total Trainable Parameters: {}".format(total_params))

if 'ft' in args.model or 'lora' in args.model:
    mark_only_lora_as_trainable(net)

net.to(device)

if args.resume:
    # Load checkpoint.
    logger.info('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.pth')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss()

def add_weight_decay(model, weight_decay=5e-4, skip_list=()):
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if len(param.shape) == 1 or name in skip_list:
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {'params': no_decay, 'weight_decay': 0.},
        {'params': decay, 'weight_decay': weight_decay}]

parameters = add_weight_decay(net, args.wd, skip_list=('bn', 'bias', 'gn'))
weight_decay = 0.
optimizer = optim.SGD(parameters, lr=args.lr,
                      momentum=0.9, weight_decay=weight_decay)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=args.epochs)
train_loss_ = []
test_loss_ = []
train_acc_ = []
test_acc_ = []
total_time = 0.0

def train(epoch):
    logger.info('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    global total_time
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        start_time = time.time()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, net.parameters()), max_norm=5)
        optimizer.step()
        torch.cuda.synchronize()
        used_time = time.time() - start_time
        total_time += used_time
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        if batch_idx % (len(trainloader)//6) == 0:
            logger.info('Train Loss: {:.3f} | Acc: {:.3f} ({:d}/{:d}) | batch_train_time: {:.3f} | total_train_time: {:.3f})'.format(
                train_loss / (batch_idx + 1), 100. * correct / total, correct, total, used_time, total_time))
        if torch.isnan(torch.tensor(loss.item())):
            return True
    train_loss_.append(train_loss / (batch_idx + 1))
    train_acc_.append(correct / total)
    return False

def test(net, testloader):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    correct_sample_indices = []
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct_sample_indices.extend(predicted.eq(targets).cpu().detach().numpy().tolist())
            correct += predicted.eq(targets).sum().item()
    test_loss_.append(test_loss / (batch_idx + 1))
    test_acc_.append(correct / total)
    if (correct / total) > best_acc:
        best_acc = correct / total
    logger.info('Test Loss: {:.3f} | Acc: {:.3f} ({:d}/{:d})'.format(
        test_loss / (batch_idx + 1), 100. * correct / total, correct, total))
    return 100. * correct / total, np.where(correct_sample_indices)[0]

for epoch in range(start_epoch, start_epoch+args.epochs):
    terminate = train(epoch)
    acc, sampler_indices = test(net, testloader)
    scheduler.step()
    logger.info('Best test accuracy so far: {:.3f}'.format(100. * best_acc))

    if terminate:
        break

torch.save(net.state_dict(), args.model_save_dir)
logger.info("Train losses: %s", train_loss_)
logger.info("Test losses: %s", test_loss_)
draw(train_loss_, test_loss_, train_acc_, test_acc_, args.picture_save_dir)

# Load pretrained weights
print('==> Loading model weights from {}'.format(args.model_save_dir))
net.load_state_dict(torch.load(args.model_save_dir))
net.to(device)

total_params = count_parameters(net)
print("Total Trainable Parameters: {}".format(total_params))
cifar10_c(args, net, args.model_save_dir)