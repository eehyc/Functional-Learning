import torch
import collections
import t_utility

class Sequential(torch.nn.Sequential):

    # def __init__(self, *args):
    #     super(Sequential, self).__init__(args)

    def sense(self, input):

        for module in self:

            if hasattr(module, 'sense'):

                input = module.sense(input)

        return input

    def forward(self, input):

        if type(input) == tuple:

            input, aux_params = input

            for module in self:

                if hasattr(module, 'forward_p'):
                    input = module.forward_p(input, aux_params)
                else:
                    input = module(input)

            return input

        else:
            for module in self:
                input = module(input)
            return input

    def requires_grad_z(self, requires_grad=True):
        for module in self:
            if hasattr(module, 'requires_grad_z'):
                module.requires_grad_z(requires_grad)

    def requires_grad_p(self, requires_grad=True):
        for module in self:
            if hasattr(module, 'requires_grad_p'):
                module.requires_grad_p(requires_grad)

class PZLayer(torch.nn.Module):

    def __init__(self):

        super(PZLayer, self).__init__()

        self.z_parameters = collections.OrderedDict()
        self.p_parameters = collections.OrderedDict()

        self.skip = False

    def requires_grad_z(self, requires_grad=True):
        for name, para in self.z_parameters.items():
            para.requires_grad = requires_grad

    def requires_grad_p(self, requires_grad=True):
        for name, para in self.p_parameters.items():
            para.requires_grad = requires_grad


    def remove_updateCOParameters(self):

        for name, parameter in self.named_parameters():
            if name in self.z_parameters:
                self.z_parameters[name] = parameter

            if name in self.p_parameters:
                self.p_parameters[name] = parameter



class View(torch.nn.Module):

    def __init__(self, shape):

        super(View, self).__init__()

        self.shape = shape

    def forward(self, x):
        return x.view(-1, *self.shape)

class Multiply(torch.nn.Module):

    def __init__(self, scale):
        super(Multiply, self).__init__()

        self.scale = scale

    def forward(self, x):
        return x * self.scale

class Clamp(torch.nn.Module):

    def __init__(self, min, max):

        super(Clamp, self).__init__()

        self.min = min
        self.max = max

    def forward(self, x):
        return torch.clamp(x, min=self.min, max=self.max)


