import torch
import torch.nn as nn
import torch.nn.functional as F


class FreqNet(nn.Module):
    def __init__(self):
        super(FreqNet, self).__init__()
        # head
        self.conv = nn.Conv1d(1, 64, 5, stride=3)
        self.batchnorm = nn.BatchNorm1d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=5, stride=2, padding=1, dilation=1, ceil_mode=False)

        # block 1
        self.conv1_1 = nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.drop1_1 = nn.Dropout(0.2)
        self.batchnorm1_1 = nn.BatchNorm1d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu_1 = nn.ReLU(inplace=True)
        self.conv2_1 = nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.batchnorm2_1 = nn.BatchNorm1d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        # block 2
        self.conv1_2 = nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.drop1_2 = nn.Dropout(0.2)
        self.batchnorm1_2 = nn.BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu_2 = nn.ReLU(inplace=True)
        self.conv2_2 = nn.Conv1d(128, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.batchnorm2_2 = nn.BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv3_2 = nn.Conv1d(64, 128, kernel_size=1, stride=2, padding=1, bias=False)
        self.batchnorm2_2 = nn.BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        # block 3
        self.conv1_3 = nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.drop1_3 = nn.Dropout(0.2)
        self.batchnorm1_3 = nn.BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu_3 = nn.ReLU(inplace=True)
        self.conv2_3 = nn.Conv1d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.batchnorm2_3 = nn.BatchNorm1d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv3_3 = nn.Conv2d(128, 256, kernel_size=1, stride=2, padding=1, bias=False)
        self.batchnorm2_3 = nn.BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        self.finalpool = nn.AdaptiveAvgPool1d(output_size=1)
        self.linear = nn.Linear(256, N, bias=False)

    def forward(self, x):
        batch_size = x.size(0)
        x = x.to(device)
        # head
        x = self.conv(x)
        x = self.batchnorm(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # block 1
        x_remember = x.clone()
        x = self.conv1_1(x)
        x = self.drop1_1(x)
        x = self.batchnorm1_1(x)
        x = self.relu_1(x)
        x = self.conv2_1(x)
        x = self.batchnorm2_1(x)
        x = x + x_remember
        x = F.relu(x)

        # block 2
        x_remember = x.clone()
        x = self.conv1_2(x)
        x = self.drop1_2(x)
        x = self.batchnorm1_2(x)
        x = self.relu_2(x)
        x = self.conv2_2(x)
        x = self.batchnorm2_2(x)
        x = x + F.pad(x_remember, [0, 0, 0, 0, 32, 32, 0, 0])
        x = F.relu(x)

        # block 3
        x_remember = x.clone()
        x = self.conv1_3(x)
        x = self.drop1_3(x)
        x = self.batchnorm1_3(x)
        x = self.relu_3(x)
        x = self.conv2_3(x)
        x = self.batchnorm2_3(x)
        x = x + F.pad(x_remember, [0, 0, 0, 0, 64, 64, 0, 0])
        x = F.relu(x)

        # end
        x = self.finalpool(x)
        #         print(x.shape)
        x = self.linear(x.view(batch_size, -1))
        #         x = F.log_softmax(x, dim=1)

        return x


class SpecNet(nn.Module):
    """
    98% on spectrograms with 3 windows
    """
    def __init__(self, N):
        super(SpecNet, self).__init__()
        self.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        # head
        self.conv = nn.Conv2d(1, 64, 5, stride=3)
        self.batchnorm = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=7, stride=3, padding=1, dilation=1, ceil_mode=False)

        # block 1
        self.conv1_1 = nn.Conv2d(64, 64, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), bias=False)
        self.drop1_1 = nn.Dropout(0.2)
        self.batchnorm1_1 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu_1 = nn.ReLU(inplace=True)
        self.conv2_1 = nn.Conv2d(64, 64, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), bias=False)
        self.batchnorm2_1 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        # block 2
        self.conv1_2 = nn.Conv2d(64, 128, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), bias=False)
        self.drop1_2 = nn.Dropout(0.2)
        self.batchnorm1_2 = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu_2 = nn.ReLU(inplace=True)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), bias=False)
        self.batchnorm2_2 = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv3_2 = nn.Conv2d(64, 128, kernel_size=(1, 1), stride=(2, 2), padding=(1, 1), bias=False)
        self.batchnorm2_2 = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        # block 3
        self.conv1_3 = nn.Conv2d(128, 256, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), bias=False)
        self.drop1_3 = nn.Dropout(0.2)
        self.batchnorm1_3 = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu_3 = nn.ReLU(inplace=True)
        self.conv2_3 = nn.Conv2d(256, 256, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), bias=False)
        self.batchnorm2_3 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv3_3 = nn.Conv2d(128, 256, kernel_size=(1, 1), stride=(2, 2), padding=(1, 1), bias=False)
        self.batchnorm2_3 = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        self.finalpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.linear = nn.Linear(256, N, bias=False)

    def forward(self, x):
        batch_size = x.size(0)
        x = x.to(self.device)
        # head
        x = self.conv(x)
        x = self.batchnorm(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # block 1
        x_remember = x.clone()
        x = self.conv1_1(x)
        x = self.drop1_1(x)
        x = self.batchnorm1_1(x)
        x = self.relu_1(x)
        x = self.conv2_1(x)
        x = self.batchnorm2_1(x)
        x = x + x_remember
        x = F.relu(x)

        # block 2
        x_remember = x.clone()
        x = self.conv1_2(x)
        x = self.drop1_2(x)
        x = self.batchnorm1_2(x)
        x = self.relu_2(x)
        x = self.conv2_2(x)
        x = self.batchnorm2_2(x)
        x = x + F.pad(x_remember, [0, 0, 0, 0, 32, 32, 0, 0])
        x = F.relu(x)

        # block 3
        x_remember = x.clone()
        x = self.conv1_3(x)
        x = self.drop1_3(x)
        x = self.batchnorm1_3(x)
        x = self.relu_3(x)
        x = self.conv2_3(x)
        x = self.batchnorm2_3(x)
        x = x + F.pad(x_remember, [0, 0, 0, 0, 64, 64, 0, 0])
        x = F.relu(x)

        # end
        x = self.finalpool(x)
        #         print(x.shape)
        x = self.linear(x.view(batch_size, -1))
        #         x = F.log_softmax(x, dim=1)

        return x


class SpecNetSmall(nn.Module):
    """
    98% on spectrograms with 3 windows
    """
    def __init__(self, N):
        super(SpecNetSmall, self).__init__()
        self.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        # head
        self.conv = nn.Conv2d(1, 64, 5, stride=3)
        self.batchnorm = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=5, stride=3, padding=1, dilation=1, ceil_mode=False)

        # block 1
        self.conv1_1 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.drop1_1 = nn.Dropout(0.2)
        self.batchnorm1_1 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu_1 = nn.ReLU(inplace=True)
        self.conv2_1 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.batchnorm2_1 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        # block 2
        self.conv1_2 = nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.drop1_2 = nn.Dropout(0.2)
        self.batchnorm1_2 = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu_2 = nn.ReLU(inplace=True)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.batchnorm2_2 = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv3_2 = nn.Conv2d(64, 128, kernel_size=(1, 1), stride=(2, 2), padding=(1, 1), bias=False)
        self.batchnorm2_2 = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        # block 3
        self.conv1_3 = nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.drop1_3 = nn.Dropout(0.2)
        self.batchnorm1_3 = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu_3 = nn.ReLU(inplace=True)
        self.conv2_3 = nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.batchnorm2_3 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv3_3 = nn.Conv2d(128, 256, kernel_size=(1, 1), stride=(2, 2), padding=(1, 1), bias=False)
        self.batchnorm2_3 = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        # block 4
        self.conv1_4 = nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.drop1_4 = nn.Dropout(0.2)
        self.batchnorm1_4 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu1_4 = nn.ReLU(inplace=True)
        self.conv2_4 = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.batchnorm2_4 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv3_4 = nn.Conv2d(256, 512, kernel_size=(1, 1), stride=(2, 2), padding=(1, 1), bias=False)
        self.batchnorm2_4 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        self.finalpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.linear = nn.Linear(512, N, bias=False)

    def forward(self, x):
        batch_size = x.size(0)
        x = x.to(self.device)
        # head
        x = self.conv(x)
        x = self.batchnorm(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # block 1
        x_remember = x.clone()
        x = self.conv1_1(x)
        x = self.drop1_1(x)
        x = self.batchnorm1_1(x)
        x = self.relu_1(x)
        x = self.conv2_1(x)
        x = self.batchnorm2_1(x)
        x = x + x_remember
        x = F.relu(x)

        # block 2
        x_remember = x.clone()
        x = self.conv1_2(x)
        x = self.drop1_2(x)
        x = self.batchnorm1_2(x)
        x = self.relu_2(x)
        x = self.conv2_2(x)
        x = self.batchnorm2_2(x)
        x = x + F.pad(x_remember, [0, 0, 0, 0, 32, 32, 0, 0])
        x = F.relu(x)

        # block 3
        x_remember = x.clone()
        x = self.conv1_3(x)
        x = self.drop1_3(x)
        x = self.batchnorm1_3(x)
        x = self.relu_3(x)
        x = self.conv2_3(x)
        x = self.batchnorm2_3(x)
        x = x + F.pad(x_remember, [0, 0, 0, 0, 64, 64, 0, 0])
        x = F.relu(x)

        # block 4
        x_remember = x.clone()
        x = self.conv1_3(x)
        x = self.drop1_3(x)
        x = self.batchnorm1_3(x)
        x = self.relu_3(x)
        x = self.conv2_3(x)
        x = self.batchnorm2_3(x)
        x = x + F.pad(x_remember, [0, 0, 0, 0, 128, 128, 0, 0])
        x = F.relu(x)

        # end
        x = self.finalpool(x)
        #         print(x.shape)
        x = self.linear(x.view(batch_size, -1))
        #         x = F.log_softmax(x, dim=1)

        return x


class ResBlock(nn.Module):
    def __init__(self, n_in, n_out, k, s, p):
        super(ResBlock, self).__init__()

        self.mini_block_1 = nn.Sequential(

        )