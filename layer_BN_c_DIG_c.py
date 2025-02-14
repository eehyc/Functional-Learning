import t_utility
import t_io
import torch
import logging
import numpy as np
import t_layer


class MyBatchNorm2d(torch.nn.BatchNorm2d):
    def __init__(self, num_features, eps=1e-5, momentum=0.1,
                 affine=True, track_running_stats=True):
        super(MyBatchNorm2d, self).__init__(
            num_features, eps, momentum, affine, track_running_stats)

    def forward(self, input):
        self._check_input_dim(input)

        exponential_average_factor = 0.0

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        # calculate running estimates
        if self.training:
            mean = input.mean([0, 2, 3])
            # use biased var in train
            var = input.var([0, 2, 3], unbiased=False)
            n = input.numel() / input.size(1)
            with torch.no_grad():
                self.running_mean = exponential_average_factor * mean\
                    + (1 - exponential_average_factor) * self.running_mean
                # update running_var with unbiased var
                self.running_var = exponential_average_factor * var * n / (n - 1)\
                    + (1 - exponential_average_factor) * self.running_var
        else:
            mean = self.running_mean
            var = self.running_var

        input = (input - mean[None, :, None, None]) / (torch.sqrt(var[None, :, None, None] + self.eps))
        if self.affine:
            input = input * self.weight[None, :, None, None] + self.bias[None, :, None, None]

        return input


class MyBatchNorm1d(torch.nn.BatchNorm1d):
    def __init__(self, num_features, eps=1e-5, momentum=0.1,
                 affine=True, track_running_stats=True):
        super(MyBatchNorm1d, self).__init__(
            num_features, eps, momentum, affine, track_running_stats)

    def forward(self, input):
        self._check_input_dim(input)

        exponential_average_factor = 0.0

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        # calculate running estimates
        if self.training:
            mean = input.mean([0])
            # use biased var in train
            var = input.var([0], unbiased=False)
            n = input.numel() / input.size(1)
            with torch.no_grad():
                self.running_mean = exponential_average_factor * mean\
                    + (1 - exponential_average_factor) * self.running_mean
                # update running_var with unbiased var
                self.running_var = exponential_average_factor * var * n / (n - 1)\
                    + (1 - exponential_average_factor) * self.running_var
        else:
            mean = self.running_mean
            var = self.running_var

        #input = (input - mean[None, :]) / (torch.sqrt(var[None, :] + self.eps))
        input = (input) / (torch.sqrt(var[None, :] + self.eps))

        if self.affine:
            input = input * self.weight[None, :] + self.bias[None, :]

        return input

class layer_BN_c_DIG_c(t_layer.PZLayer):


    def __init__(self, channel, name = ''):

        super(layer_BN_c_DIG_c, self).__init__()

        self.name = name
        self.channel = channel
        self.skip = False

        self.initParams()

        self.debug = False



    def initParams(self):
        #
        #
        self.register_buffer('running_max', torch.ones(3) * 0.1)
        self.register_buffer('momentum', torch.ones(1) * 0.01)

        # mean = torch.mean(outputs, dim=1)


        # self.direction = torch.nn.Parameter(torch.rand([3, 32, 32]))
        # self.p_parameters['direction'] = self.direction
        # self.offset = torch.nn.Parameter(torch.zeros(3))
        # self.p_parameters['offset'] = self.offset

        self.bn = torch.nn.BatchNorm1d(num_features=32*32*3, affine=False, momentum=0.01)
        # self.bn = MyBatchNorm1d(num_features=32 * 32 * 3, affine=False, momentum=0.01)
        # self.bn = torch.nn.BatchNorm2d(num_features=3, affine=False, momentum=0.01)

        for name, par in self.bn.named_parameters():
            self.p_parameters['bn.'+name]=par

        # self.scale = torch.nn.Parameter(torch.ones(1))
        # self.p_parameters['scale'] = self.scale


    def forward(self, inputs):

        # p = np.random.rand()
        # if p < 0.01:
        #     self.debug = True
        # else:
        #     self.debug = False

        if self.skip == True:
            return inputs

        outputs = inputs


        # outputs = torch.sum(inputs,dim=1,keepdim=True)

        # direction
        # outputs = outputs * self.direction + self.offset

        # if self.debug == True:
        #     with torch.no_grad():
        #         print('input', outputs.mean(dim=[0,2,3]).cpu().numpy(), outputs.std(dim=[0,2,3]).cpu().numpy())

        # cross operation
        # def Patch(img):
        #
        #     up, down = torch.chunk(img, 2, dim=2)
        #
        #     up_left, up_right = torch.chunk(up, 2, dim=3)
        #     down_left, down_right = torch.chunk(down, 2, dim=3)
        #
        #     return up_left, up_right, down_left, down_right
        #
        # up_left, up_right, down_left, down_right = Patch(outputs)
        #
        # up = torch.cat((up_left - down_right, up_right - down_left), dim=3)
        # down = torch.cat((down_left - up_right, down_right - up_left), dim=3)
        # # up = torch.cat((up_left - down_right.flip([2,3]), up_right - down_left.flip([2,3])), dim=3)
        # # down = torch.cat((down_left - up_right.flip([2,3]), down_right - up_left.flip([2,3])), dim=3)
        # outputs = torch.cat((up, down), dim=2)


        outputs = outputs - outputs.flip(2).flip(3)


        # if self.debug == True:
        #     with torch.no_grad():
        #         print('x-', outputs.mean(dim=[0,2,3]).cpu().numpy(), outputs.std(dim=[0,2,3]).cpu().numpy())


        # if self.training == True:
        #     with torch.no_grad():
        #         for c in range(self.channel):
        #             mmx = torch.max(outputs[:, c, ...])
        #             # mmx = torch.std(outputs[:,c,...])
        #             a = self.momentum * (self.momentum * (mmx - self.running_max[c]))
        #             self.running_max[c] = a + self.running_max[c]
        #
        # for c in range(self.channel):
        #     outputs[:,c,...] = outputs[:,c,...] / (self.running_max[c])

        sp = outputs.shape
        outputs = outputs.view(sp[0], -1)
        outputs = self.bn(outputs)

        # if self.debug == True:
        #     print(self.bn.running_mean)
        #     print(self.bn.running_var)
        #     print(1)

        # mean = torch.mean(outputs)
        # var = torch.var(outputs)
        # outputs = (outputs - mean) / (var)

        outputs = outputs.view(sp)

        # if self.debug == True:
        #     with torch.no_grad():
        #         print('running max', self.running_max)
        #         print('normalize-', outputs.mean(dim=[0,2,3]).cpu().numpy(),
        #               outputs.std(dim=[0,2,3]).cpu().numpy(),
        #               outputs.max().cpu().numpy())


        outputs = torch.clamp(outputs, min=0.0, max=1.0)

        # outputs = outputs.repeat(1,3,1,1)

        # if self.debug == True:
        #     with torch.no_grad():
        #         print('final ', outputs.mean(dim=[0,2,3]).cpu().numpy())

        return outputs



    def sense(self, inputs):

        if self.skip == True:
            return inputs

        # device = self.bn.running_mean.device
        device = 'cuda'

        self.eval()
        self.debug = True
        with torch.no_grad():
            outputs = self.forward(t_io.ImgConvertTorchAndNumpy(inputs).to(device))
        self.debug = False


        outputs = t_io.ImgConvertTorchAndNumpy(outputs)

        return outputs


