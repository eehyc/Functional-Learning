import torch
import torch.nn as nn
import t_layer

import numpy as np

import t_utility
import logging
import time


class oLayer_FC_i_xLT_i(t_layer.PZLayer):

    def __init__(self, oDevice, params, FLAGS, name = ''):

        super(oLayer_FC_i_xLT_i, self).__init__()

        self.flag = False

        self.is_OLayer = True

        self.name = name
        self.oDevice = oDevice
        self.channel = oDevice.channel
        self.grain = oDevice.spec['grain']
        self.LCCount = oDevice.spec['LCCount']

        self.postprocess = True

        self.noise_level = 0
        self.energy_scale = 10
        self.malfunction_ratio = 0
        self.initParams()

        self.l1_reg = 0


    def initParams(self):



        self.register_buffer('noise_std', torch.ones(1) * 0.0001)

        self.p_updates = []
        self.z_updates = []

        # z part
        z_parameters = self.z_parameters
        spec = self.oDevice.spec

        for c in range(len(self.channel)):

            if self.channel[c] == 0:
                continue

            for l in range(spec['LCCount']):

                key = 'LC' + str(l) + 'ScopicWeight_c' + str(c)
                setattr(self, key, nn.Parameter(torch.ones([len(self.grain), spec['oVirtualWidthP'] * spec['oVirtualHeightP'] * spec['iVirtualWidthP'] * spec['iVirtualHeightP']])))

                z_parameters[key] = getattr(self, key)

            key = 'NLScopicWeight_c' + str(c)
            setattr(self, key, nn.Parameter(torch.ones([len(self.grain), spec['oVirtualWidthP'] * spec['oVirtualHeightP'] * spec['iVirtualWidthP'] * spec['iVirtualHeightP']])))
            z_parameters[key] = getattr(self, key)


        # key = 'OutputGain_c' + str(c)
        key = 'OutputLever'
        setattr(self, key, nn.Parameter(torch.ones([1])))
        z_parameters[key] = getattr(self, key)

        if self.postprocess == True:

            self.postprocess = nn.Sequential(
                nn.Conv2d(np.sum(self.channel), 64, 3, padding=1, bias=True),
                nn.ReLU(),
                nn.Conv2d(64, 64, 3, padding=1, bias=True),
                nn.ReLU(),
                nn.Conv2d(64, 64, 3, padding=1, bias=True),
                nn.ReLU(),
                nn.Conv2d(64, 64, 3, padding=1, bias=True),
                nn.ReLU(),
                nn.Conv2d(64, np.sum(self.channel), [3, 3], padding=1, bias=False))

            for name, value in self.postprocess.named_parameters():
                z_parameters['postprocess.' + name] = value

        # Operation
        p_parameters = self.p_parameters
        for c in range(len(self.channel)):

            if self.channel[c] == 0:
                continue

            for l in range(spec['LCCount']):
                key = 'LC' + str(l) + '_c' + str(c)
                # setattr(self, key, nn.Parameter(torch.ones([1, spec['lVirtualWidthP'] * spec['lVirtualHeightP']])))
                setattr(self, key, nn.Parameter(torch.rand([1, spec['lVirtualWidthP'] * spec['lVirtualHeightP']])))
                p_parameters[key] = getattr(self, key)

                self.p_updates.append(t_utility.Constrain(getattr(self, key), 0.0, 1.0))

                key = key + '_mask'
                init_mask = torch.rand([1, spec['lVirtualWidthP'] * spec['lVirtualHeightP']])
                init_mask[init_mask < self.malfunction_ratio] = 0
                init_mask[init_mask > 0] = 1
                init_mask = 1.0 - init_mask

                self.register_buffer(key, init_mask)


            key = 'InputAttention_c' + str(c)
            setattr(self, key, nn.Parameter(torch.ones([spec['iVirtualHeightP'], spec['iVirtualWidthP']])))
            p_parameters[key] = getattr(self, key)
            self.p_updates.append(t_utility.Constrain(getattr(self, key), 1.0 / 255.0, 1.0))

            key = 'OutputAttention_c' + str(c)
            setattr(self, key, nn.Parameter(torch.ones([spec['oVirtualHeightP'], spec['oVirtualWidthP']])))
            p_parameters[key] = getattr(self, key)
            self.p_updates.append(t_utility.Constrain(getattr(self, key), 1.0, 255.0))

        assert len(p_parameters) + len(z_parameters) == len(list(self.parameters())), 'The numbers of parameters are not equal.'


        self.o_parameters = self.p_parameters
        self.c_parameters = self.z_parameters
        self.o_updates = self.p_updates
        self.c_updates = self.z_updates

        return

    def forward(self, inputs):

        return self.forward_p(inputs, self.p_parameters)


    def forward_p(self, inputs, p_parameters):

        if self.skip == True:
            return inputs

        num_lc = self.oDevice.spec['LCCount']
        num_channel = int(np.sum(self.channel))

        LCs = {}
        for l in range(num_lc):
            for c in range(num_channel):

                if self.channel[c] == 0:
                    continue

                key = 'LC' + str(l) + '_c' + str(c)
                LCs[key] = p_parameters[key]

                if self.malfunction_ratio > 0:
                    mask_key = key + '_mask'
                    LCs[key] = torch.max(p_parameters[key], getattr(self, mask_key))


        input_attention = t_utility.mergeParameters(p_parameters, 'InputAttention_c%i', range(num_channel), 0)
        output_attention = t_utility.mergeParameters(p_parameters, 'OutputAttention_c%i', range(num_channel), 0)



        outputs = self.kernel(inputs, LCs, input_attention, output_attention)


        return outputs



    def kernel_ScopicLayer(self, lc_value, scopic_weight, prefix, c, grain_s, propagations, l_hw, i_hw, o_hw):

        for g in range(0, len(grain_s)):

            grain = grain_s[g]

            if not lc_value is None:

                lc_transmission = propagations[prefix + 'Propagation_c' + str(c) + '_g' + str(grain)]

                lc = t_utility.DownScale(lc_value, (1, 1, grain, grain))
                lc = lc.view((-1, (l_hw[0] * l_hw[1]) // (grain * grain)))

                lc_transmission = lc_transmission.matmul(lc,transpose_b=True).t()  # b, grain oh * grain ow * grain ih * grain iw
                # lc_transmission = lc_transmission.t()

            else:
                if g == 0:
                    lc_transmission = propagations[prefix + 'Propagation_c' + str(c) + '_g' + str(grain)]
                else:
                    continue

            lc_transmission = lc_transmission.view((-1, o_hw[0] // grain, o_hw[1] // grain, i_hw[0] // grain, i_hw[1] // grain))

            lc_transmission = t_utility.UpScale(lc_transmission, [1, grain, grain, grain, grain])

            lc_transmission = lc_transmission.view(
                (-1, o_hw[0] * o_hw[1] * i_hw[0] * i_hw[1]))  # (oh * ow * ih * iw)

            lc_transmission = torch.mul(lc_transmission, scopic_weight[g, :])  # b, (oh * ow * ih * iw)


            if g == 0:
                lc_transmission_grain = lc_transmission
            else:
                lc_transmission_grain += lc_transmission

        return lc_transmission_grain


    def kernel_per_channel(self, grain_s, propagations, LCs, l_hw, i_hw, o_hw, spec, c, inputs):

        for l in range(0, spec['LCCount']):

            lc_value = LCs['LC' + str(l) + '_c' + str(c)]

            lc_value = lc_value.view((-1, 1, l_hw[0], l_hw[1]))

            lc_scopic_weight = self.z_parameters['LC' + str(l) + 'ScopicWeight' + '_c' + str(c)]

            if l == 0:
                t_trainsmission = self.kernel_ScopicLayer(lc_value, lc_scopic_weight, 'LC' + str(l), c, grain_s, propagations, l_hw, i_hw, o_hw)
            else:
                lcx_trainsmission = self.kernel_ScopicLayer(lc_value, lc_scopic_weight, 'LC' + str(l), c, grain_s, propagations, l_hw, i_hw, o_hw)
                t_trainsmission = torch.mul(t_trainsmission, lcx_trainsmission)


        # NL
        nl_scopic_weight = self.z_parameters['NLScopicWeight' + '_c' + str(c)]

        nlp = self.kernel_ScopicLayer(None, nl_scopic_weight, 'NL', c, grain_s, propagations, l_hw, i_hw, o_hw)


        t_trainsmission = torch.mul(nlp, t_trainsmission)

        t_trainsmission = t_trainsmission.view((-1, o_hw[0] * o_hw[1], i_hw[0] * i_hw[1]))


        if t_trainsmission.size()[0] == 1:  # only one LC

            inputs = inputs.view((-1, i_hw[0] * i_hw[1]))

            o = torch.mm(inputs, torch.squeeze(t_trainsmission).t())

        else:  # has a batch of LC

            inputs = inputs.view((-1, 1, i_hw[0] * i_hw[1]))
            o = torch.bmm(inputs, t_trainsmission.transpose(1, 2))


        return o

    def kernel(self, inputs, LCs, input_attention, output_attention):

        spec = self.oDevice.spec

        num_channel = inputs.size()[1]
        channel = self.channel

        inputs = torch.mul(inputs, input_attention)
        input_split = torch.chunk(inputs, num_channel, 1)

        propagations = self.oDevice.propagationTensors

        grain_s = self.grain

        o_hw = (spec['oVirtualHeightP'], spec['oVirtualWidthP'])
        i_hw = (spec['iVirtualHeightP'], spec['iVirtualWidthP'])
        l_hw = (spec['lVirtualHeightP'], spec['lVirtualWidthP'])

        outputs = []
        in_ptr = 0
        for c in range(len(channel)):

            if channel[c] == 0:
                continue

            o = self.kernel_per_channel(grain_s, propagations, LCs, l_hw, i_hw, o_hw, spec, c, input_split[in_ptr])
            o = o.view((-1, o_hw[0], o_hw[1]))

            outputs.append(o)
            in_ptr += 1


        outputs = torch.stack(outputs, 1)


        if self.postprocess:
            outputs = self.postprocess(outputs)


        output_lever = self.z_parameters['OutputLever']
        outputs = torch.mul(outputs, output_lever)


        outputs = torch.mul(outputs, output_attention)


        if self.l1_reg > 0:
            self.layer_loss = self.l1_reg * (torch.norm(outputs, 1))

        return outputs





    def sense(self, inputs, LCs=None, input_attention=None, output_attention=None):

        # print('flag ', self.flag)

        if self.skip == True:
            return inputs

        # prepare parameters
        with torch.no_grad():
            if LCs is None:
                lc0 = t_utility.mergeParameters(self, 'LC0_c%i', range(3), -1).detach().cpu().numpy()
                lc1 = t_utility.mergeParameters(self, 'LC1_c%i', range(3), -1).detach().cpu().numpy()
                LCs = [lc0, lc1]

            if input_attention is None:
                input_attention = t_utility.mergeParameters(self, 'InputAttention_c%i', range(3), -1).detach().cpu().numpy()

            if output_attention is None:
                output_attention = t_utility.mergeParameters(self, 'OutputAttention_c%i', range(3), -1).detach().cpu().numpy()


        if self.malfunction_ratio > 0:
            for l in range(self.oDevice.spec['LCCount']):
                key = 'LC' + str(l) + '_c%i_mask'
                mask = t_utility.mergeParameters(self, key, range(3), -1).detach().cpu().numpy()

                LCs[l] = np.maximum(mask, LCs[l])


        # send to remote server if client exist
        if self.flag == False and not self.oDevice.client == None:

            client = self.oDevice.client

            send_data = {}

            send_data['LCs'] = LCs
            send_data['input_attention'] = input_attention
            send_data['output_attention'] = output_attention

            connections = []
            try:
                # this might have bugs
                if inputs.ndim >= 4 and inputs.shape[0] > client.suggested_batch_size: # batched data, consider caching


                    b = inputs.shape[0]
                    i = 0
                    batch_size = int(client.suggested_batch_size)
                    interval = client.suggested_batch_interval
                    while i < b:

                        send_data['inputs'] = inputs[i:min(i + batch_size, b),...]
                        connections.append(client.connect(send_data.copy()))
                        i += batch_size
                        time.sleep(interval)

                    outputs = []
                    for i in range(len(connections)):
                        while connections[i].isAlive():
                            time.sleep(1)
                        outputs.append(connections[i].result()['outputs'])

                    outputs = np.concatenate(outputs, axis=0) * self.energy_scale

                else:


                    send_data['inputs'] = inputs
                    connections.append(client.connect(send_data))
                    while connections[0].is_alive():
                        time.sleep(1)

                    recv_data = connections[0].result()['outputs']

                    outputs = recv_data * self.energy_scale

            except BaseException as e:
                logging.error("Failed receving from server, please check your connection")
                logging.error(e)
                for connection in connections:
                    connection.shutdown()
                raise TimeoutError
                outputs = None


        return outputs


