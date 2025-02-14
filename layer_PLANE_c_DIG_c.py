import t_utility
import torch
import logging
import numpy as np
import t_layer



class layer_PLANE_c_DIG_c(t_layer.PZLayer):


    def __init__(self, channel, name = ''):

        super(layer_PLANE_c_DIG_c, self).__init__()

        self.name = name
        self.channel = channel
        self.skip = False

        self.regular = True

        self.initParams()



    def initParams(self):

        self.updates = []

        self.scale = torch.nn.Parameter(torch.ones(self.channel))

        self.p_parameters['scale'] = self.scale

        # self.add_module('dropout', torch.nn.Dropout(p=0.5))



    def forward(self, inputs):

        outputs = []

        self.layer_loss = 0

        for c in range(self.channel):
            outputs.append(inputs[:,c,...] * self.scale[c])
            # if self.regular == True:
            #     self.layer_loss = (torch.max(outputs[-1]) - 1.0) ** 2
            if self.regular == True:
                outputs[-1] = torch.clamp(outputs[-1], 0.0, 1.0)
                self.layer_loss += (torch.max(outputs[-1]) - 1.0) ** 2

        outputs = torch.stack(outputs, dim=1)

        return outputs

    def sense(self, inputs):

        if self.skip == True:
            return inputs

        scale = self.scale.detach().cpu().numpy()
        inputs = inputs * scale
        print('the gain is ', scale)
        outputs = np.clip(inputs, 0.0, 1.0)


        return outputs


