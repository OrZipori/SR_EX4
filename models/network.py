import torch
import torch.nn as nn
import torch.nn.functional as F

class ProbNet(nn.Module):
    def __init__(self, lstm_out, num_classes, first_filter, second_filter, batch_size):
        super(ProbNet, self).__init__()
        self.conv1 = nn.Conv2d(1, first_filter, kernel_size=3)
        self.conv2 = nn.Conv2d(first_filter, second_filter, kernel_size=3)
    
        self.conv_batch_norm1 = nn.BatchNorm2d(first_filter)
        self.conv_batch_norm2 = nn.BatchNorm2d(second_filter)
        
        self.conv_drop1 = nn.Dropout(p=0.25)
        self.conv_drop2 = nn.Dropout(p=0.5)

        # calculate the end dimensions after 2 convolution blocks
        new_h, new_w = self.conv_output_shape((161, 101), kernel_size=3)
        new_h, new_w = self.conv_output_shape((new_h, new_w), kernel_size=2, stride=2)
        new_h, new_w = self.conv_output_shape((new_h, new_w), kernel_size=3)
        new_h, new_w = self.conv_output_shape((new_h, new_w), kernel_size=2, stride=2)

        lstm_in = new_h * second_filter
        self.linear2 = nn.Linear(lstm_out, num_classes)

        self.log_softmax = nn.LogSoftmax(dim=2)

        self.second_filter = second_filter
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.lstm_out = lstm_out

        self.lstm = nn.LSTM(lstm_in, lstm_out)

    def conv_output_shape(self, h_w, kernel_size=1, stride=1, pad=0, dilation=1):
        # https://discuss.pytorch.org/t/utility-function-for-calculating-the-shape-of-a-conv-output/11173/4 DuaneNielsen
        from math import floor
        if type(kernel_size) is not tuple:
            kernel_size = (kernel_size, kernel_size)
        h = floor( ((h_w[0] + (2 * pad) - ( dilation * (kernel_size[0] - 1) ) - 1 )/ stride) + 1)
        w = floor( ((h_w[1] + (2 * pad) - ( dilation * (kernel_size[1] - 1) ) - 1 )/ stride) + 1)
        return h, w

    def forward(self, x, device):
        # first conv - relu - pool block
        out = self.conv1(x)
        out = F.relu(out)
        out = self.conv_batch_norm1(out)
        out = self.conv_drop1(out)
        out = F.max_pool2d(out, 2)

        # second conv - relu - pool block
        out = self.conv2(out)
        out = F.relu(out)
        out = self.conv_batch_norm2(out)
        out = self.conv_drop2(out)
        out = F.max_pool2d(out, 2)

        batch_size, channels, freq, frames = out.size()
        out = out.view(batch_size,  channels *  freq, frames)
        out = out.permute(2, 0, 1)

        out, _ = self.lstm(out)

        out = out.permute(1, 0, 2)

        out = self.linear2(out)

        out = self.log_softmax(out)

        out = out.permute(1, 0, 2)
        return out
