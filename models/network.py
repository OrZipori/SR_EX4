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

        # calculate the end dimensions after 2 convolution blocks
        new_h, new_w = self.conv_output_shape((161, 101), kernel_size=3)
        new_h, new_w = self.conv_output_shape((new_h, new_w), kernel_size=2, stride=2)
        new_h, new_w = self.conv_output_shape((new_h, new_w), kernel_size=3)
        new_h, new_w = self.conv_output_shape((new_h, new_w), kernel_size=2, stride=2)

        lstm_in = 600
        self.linear1 = nn.Linear(new_h * second_filter, lstm_in)
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
        
    def init_hidden(self, batch_size, device):
        rnn_init_h = nn.Parameter(torch.zeros(1, batch_size, self.lstm_out).type(torch.FloatTensor).to(device), requires_grad=True)
        rnn_init_c = nn.Parameter(torch.zeros(1, batch_size, self.lstm_out).type(torch.FloatTensor).to(device), requires_grad=True)
        return (rnn_init_h, rnn_init_c)

    def forward(self, x, device):
        # first conv - relu - pool block
        out = self.conv1(x)
        out = self.conv_batch_norm1(out)
        out = F.relu(out)
        out = F.max_pool2d(out, 2)

        # second conv - relu - pool block
        out = self.conv2(out)
        out = self.conv_batch_norm2(out)
        out = F.relu(out)
        out = F.max_pool2d(out, 2)

        freq = out.size()[2]
        frames = out.size()[3]
        batch_size = out.size()[0]
        out = out.view(frames * batch_size, self.second_filter * freq)

        out = self.linear1(out)
        
        out = out.view(frames, batch_size, -1)
        
        #out = out.permute(3, 0, 1, 2)

        #out = out.view(frames, batch_size, self.second_filter * freq)
    
        self.h0, self.c0 = self.init_hidden(batch_size, device)

        out, _ = self.lstm(out, (self.h0, self.c0))

        out = out.view(-1, self.lstm_out)
        out = self.linear2(out)

        out = out.view(frames, batch_size, self.num_classes)

        out = self.log_softmax(out)
        return out


# increase channels to 30?
# num of freq and time frames decrease

# for lstm maps * freq 
# hidden 200 
# linear to num classes