import t_utility
import torch
import logging
import t_layer
import numpy as np



class layer_PRUNE_i_DIG_oh(t_layer.PZLayer):


    def __init__(self, input_size, out_onehot_size, grain, params, FLAGS, name = ''):

        super(layer_PRUNE_i_DIG_oh, self).__init__()

        self.inputSize = input_size
        self.ohSize = out_onehot_size
        self.grain = grain
        self.name = name

        self.target_oh = 4

        self.skip = False

        self.initParams()



    def initParams(self):


        self.register_buffer('prune_step', torch.ones(1, dtype=torch.long))

        mChannel, oHeightP, oWidthP = self.inputSize

        mHeightP = oHeightP // self.grain[0]
        mWidthP = oWidthP // self.grain[1]

        self.updates = []

        grain = self.grain

        assert(oHeightP % grain[0] == 0 and oWidthP % grain[1] == 0)

        self.dotSize = mHeightP * mWidthP


        for o in range(self.ohSize):
            key = 'propagation_oh_' + str(o)
            setattr(self, key, torch.nn.Parameter(torch.rand(self.dotSize, 1)))
            self.updates.append(t_utility.Constrain(getattr(self, key), 0., 1.))

            key = 'mask_oh_' + str(o)
            self.register_buffer(key, torch.ones(self.dotSize, 1))

        self.inputGain = torch.nn.Parameter(torch.ones(1))

        for name, par in self.named_parameters():
            self.p_parameters[name] = par


        self.o_parameters = self.p_parameters

    def resetParams(self):

        with torch.no_grad():
            for o in range(self.ohSize):
                key = 'propagation_oh_' + str(o)

                getattr(self, key).copy_(torch.randn(self.dotSize, 1))

    def setupSparsityFunction(self, sparsity_function):

        self.sparsity_function = sparsity_function

    def updatePrune(self):

        self.prune_step.add_(1)

        def Prune(p, mask, amount):

            tensor_size = p.nelement()
            nparams_toprune = int(round(amount * tensor_size))

            topk = torch.topk(torch.abs(torch.mul(p, mask)).view(-1), k=nparams_toprune, largest=False)

            mask.fill_(1.)
            # mask = torch.ones_like(p, device=p.device)
            mask.view(-1)[topk.indices] = 0


        if hasattr(self, 'sparsity_function'):
            if self.sparsity_function.Valid(self.prune_step):
                ts = self.sparsity_function(self.prune_step).item()
                logging.debug('Prune to target sparsity : %f.' % (ts))
                for o in range(self.ohSize):
                    Prune(getattr(self, 'propagation_oh_' + str(o)), getattr(self, 'mask_oh_' + str(o)), ts)


    def forward(self, inputs):

        if self.skip == True:
            return inputs

        #inputs = inputs * self.inputGain
        #inputs = t_utility.DownScale(inputs, [1, 3, self.grain[0], self.grain[1]])
        inputs = t_utility.DownScale(inputs, [1, 3, self.grain[0], self.grain[1]]) * self.inputGain

        # self.layer_loss = 0
        inputs = inputs.view(-1, self.dotSize)
        outputs_oh = []
        for o in range(self.ohSize):
            # selection = torch.mul(getattr(self, 'propagation_oh_' + str(o)), getattr(self, 'mask_oh_' + str(o)))
            # # self.layer_loss += torch.abs(torch.sum(torch.abs(selection))-self.target_oh)
            # outputs_oh.append(torch.mm(inputs, selection))

            outputs_oh.append(torch.mm(inputs, torch.mul(getattr(self, 'propagation_oh_' + str(o)), getattr(self, 'mask_oh_' + str(o)))))

        outputs = torch.cat(outputs_oh, -1)

        return outputs

    def drawMask(self, class_ids):

        final_mask = []
        for l in range(len(class_ids)):

            class_id = class_ids[l,...]

            mask = torch.ones(self.inputSize)
            small_mask = t_utility.DownScale(mask, [3, self.grain[0], self.grain[1]])
            before_shape = small_mask.shape

            my_mask_oh = getattr(self, 'mask_oh_' + str(class_id)).cpu()

            small_mask = torch.mul(small_mask.view(my_mask_oh.shape), my_mask_oh)
            small_mask = small_mask.view(before_shape)

            final_mask.append(t_utility.UpScale(small_mask, [3, self.grain[0], self.grain[1]]))

        final_mask = torch.stack(final_mask, dim = 0)

        return final_mask

    def drawWeight(self):

        final_mask = []

        for o in range(self.ohSize):

            mask = torch.ones(self.inputSize)
            small_mask = t_utility.DownScale(mask, [3, self.grain[0], self.grain[1]])
            before_shape = small_mask.shape

            my_mask_oh = torch.mul(getattr(self, 'propagation_oh_' + str(o)), getattr(self, 'mask_oh_' + str(o))).cpu()

            small_mask = torch.mul(small_mask.view(my_mask_oh.shape), my_mask_oh)
            small_mask = small_mask.view(before_shape)

            final_mask.append(t_utility.UpScale(small_mask, [3, self.grain[0], self.grain[1]]))

        output = final_mask[0]
        for i in range(1, self.ohSize):
            output = torch.max(output, final_mask[i])

        return output