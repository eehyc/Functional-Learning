import multiprocessing
import sys
import time

import math
import numpy as np
import torch


import scipy.sparse as sparse
from itertools import repeat
import t_utility
import os
import t_partitioned_sparse
import t_io
import t_component_interface
import cv2
import logging
import t_socket


def MeasurementSpec(params, FLAGS, name, inputPlane, LCs, outputPlane):

    spec = {}

    spec['channel'] = FLAGS.channel

    spec['filterPropagation'] = False
    spec['measureSamples'] = 1
    spec['threshold'] = [0.2,0.1,0.0]

    spec['grain'] = [1]

    # use multi-resolution propagation to enhance numerical stability. No use anymore
    # spec['grain'] = [1,2,4]

    spec['name'] = name
    spec['mode'] = 'Measurement'

    spec['iComponent'] = inputPlane
    spec['iVirtualHeightP'] = 32
    spec['iVirtualWidthP'] = 32

    spec['lComponent'] = LCs
    spec['LCCount'] = len(LCs)

    spec['lVirtualHeightP'] = 32
    spec['lVirtualWidthP'] = 32

    spec['oComponent'] = outputPlane
    spec['oVirtualHeightP'] = 32
    spec['oVirtualWidthP'] = 32


    return spec



# i_2L_i = image plane -> 2 LCD layers -> image plane
class ODevice_i_2L_i:

    def __init__(self, spec, params, FLAGS):

        self.module_name = 'device_i_2L_i'

        self.spec = spec

        self.name = spec['name']

        self.host_param = True

        self.partitioner = None

        self.channel = spec['channel']

        self.one_channel_propagation = 1

        self.client = None

        return

    class ComponentInterface(t_component_interface.ImageComponentInterface):

        def __init__(self, upstream, cropHeight, cropWidth, grainHeight, grainWidth):
            self.bg = None
            self.upstream = upstream
            self.upstreamHwc = upstream.hwc()
            self.cropHeight = cropHeight
            self.cropWidth = cropWidth
            self.grainHeight = grainHeight
            self.grainWidth = grainWidth
            # self.bg = bg[self.cropHeight[0]:self.cropHeight[1], self.cropWidth[0]:self.cropWidth[1],:]

        def hwc(self):
            return (int((self.cropHeight[1] - self.cropHeight[0]) / self.grainHeight),
                    int((self.cropWidth[1] - self.cropWidth[0]) / self.grainWidth), 3)

        # Read output plane.
        def sense(self):
            img = self.upstream.sense()
            img = t_io.ToHDR(img)

            img = img[self.cropHeight[0]:self.cropHeight[1], self.cropWidth[0]:self.cropWidth[1], :]

            img = t_utility.DownScale(img, [self.grainHeight, self.grainWidth, 1])
            if not self.bg is None:
                img -= self.bg


            return img

        # Control input plane.
        def emit(self, img):

            img = t_utility.UpScale(img, [self.grainHeight, self.grainWidth, 1])

            img = np.pad(img, ((self.cropHeight[0], self.upstreamHwc[0] - self.cropHeight[1]),
                               (self.cropWidth[0], self.upstreamHwc[1] - self.cropWidth[1]), (0, 0)), 'constant')
            self.upstream.emit(img)

    # Read the propagation from files.
    def readPropagation(self, location, NL_grain, LC_grain_s, scopic_weight, channel):


        logging.debug(self.module_name + ": Trying to load propagation file from: " + str(location))

        for c in range(len(channel)):

            if channel[c] == 0:
                continue

            for grain in NL_grain:
                key = 'NLPropagation' + '_c' + str(c) + '_g' + str(grain)

                file = location + key + '.npy'
                if os.path.exists(file):
                    self.propagation[key] = np.load(file)
                else:
                    self.propagation[key] = None

        LC_count = len(LC_grain_s)
        for l in range(LC_count):
            grain_s = LC_grain_s[l]
            for grain in grain_s:
                 for c in range(len(channel)):
                    if channel[c] == 0:
                        continue
                    key = 'LC' + str(l) + 'Propagation_c' + str(c) + '_g' + str(grain)
                    file = location + key + '.npz'
                    if os.path.exists(file):
                        self.propagation[key] = sparse.load_npz(file)
                    else:
                        self.propagation[key] = None



    # Record the propagation.
    def writePropagation(self, location, NL_grain, LC_grain_s, scopic_weight, channel):

        print(self.module_name + ": Writing propagation to:", location)


        for c in range(len(channel)):
            if channel[c] == 0:
                continue
            if self.one_channel_propagation >= 0 and not c == self.one_channel_propagation:
                continue
            for grain in NL_grain:
                np.save(location + 'NLPropagation_c' + str(c) + '_g' + str(grain),
                        self.propagation['NLPropagation_c' + str(c) + '_g' + str(grain)])

        LC_count = len(LC_grain_s)
        for l in range(LC_count):
            grain_s = LC_grain_s[l]
            for grain in grain_s:
                for c in range(len(channel)):
                    if channel[c] == 0:
                        continue
                    if self.one_channel_propagation >= 0 and not c == self.one_channel_propagation:
                        continue
                    sparse.save_npz(location + 'LC' + str(l) + 'Propagation_c' + str(c) + '_g' + str(grain),
                                    self.propagation['LC' + str(l) + 'Propagation_c' + str(c) + '_g' + str(grain)])


    def activateHardware(self, params, FLAGS):

        activateHardware = FLAGS.activateHardware
        spec = self.spec

        if activateHardware:

            devices = params['components']
            self.inputPlane = devices[spec['iComponent']]

            self.LCs = []
            for lc in spec['lComponent']:
                self.LCs.append(devices[lc])

            self.outputPlane = devices[spec['oComponent']]

            # test exposure
            self.inputPlane.emit(np.ones(self.inputPlane.hwc()) * self.channel)
            for lc in self.LCs:
                lc.emit(np.ones(lc.hwc()) * self.channel)
            self.inputPlane.delay()
            max_output = t_io.ToHDR(self.outputPlane.sense() * self.channel)

            logging.debug('Maximum reading digit of ' + self.module_name + self.name + ': ' + str(max_output.max()))


            # calculate the virtual resolution
            crop_height_p, crop_width_p, scalability_height_p, scalability_width_p = CalibrateCrop(spec,
                                                                                                   self.inputPlane, 'i')

            self.inputPlane = self.ComponentInterface(self.inputPlane, crop_height_p, crop_width_p,
                                                      scalability_height_p, scalability_width_p)

            new_LCs = []
            for lc in self.LCs:
                crop_height_p, crop_width_p, scalability_height_p, scalability_width_p = CalibrateCrop(spec, lc, 'l')
                new_LCs.append(
                    self.ComponentInterface(lc, crop_height_p, crop_width_p, scalability_height_p, scalability_width_p))

            self.LCs = new_LCs

            # output
            crop_height_p, crop_width_p, scalability_height_p, scalability_width_p = CalibrateCrop(spec,
                                                                                                   self.outputPlane,
                                                                                                   'o')
            self.outputPlane = self.ComponentInterface(self.outputPlane, crop_height_p, crop_width_p,
                                                       scalability_height_p, scalability_width_p)



            # BG
            self.inputPlane.emit(np.zeros(self.inputPlane.hwc()))
            for lc in self.LCs:
                lc.emit(np.zeros(lc.hwc()))
            self.inputPlane.delay()
            bg = t_io.ToHDR(self.outputPlane.sense())

            self.outputPlane.bg = bg


            # full-power output
            self.inputPlane.emit(np.ones(self.inputPlane.hwc()) * self.channel)
            for lc in self.LCs:
                lc.emit(np.ones(lc.hwc()) * self.channel)
            self.inputPlane.delay()
            self.fullPowerOutput = t_io.ToHDR(self.outputPlane.sense() * self.channel)

            logging.debug('Full Power Output of ' + self.module_name + self.name + ': ' + str(self.fullPowerOutput.max()))
        else:
            self.inputPlane = t_component_interface.DummyInterface(spec['iVirtualHeightP'], spec['iVirtualWidthP'], len(self.channel))
            self.outputPlane = t_component_interface.DummyInterface(spec['oVirtualHeightP'], spec['oVirtualWidthP'], len(self.channel))
            self.LCs = []
            for l in range(spec['LCCount']):
                self.LCs.append(
                    t_component_interface.DummyInterface(spec['lVirtualHeightP'], spec['lVirtualWidthP'], len(self.channel)))

            self.fullPowerOutput = np.ones(self.outputPlane.hwc(), dtype=np.float32)

            if FLAGS.hardwareSocket == 'client':
                self.client = t_socket.ThreadingSocket('client', FLAGS.hardwareAddress, FLAGS.hardwarePort)

    # Propagation is the (measured) coarse connections between neurons for initializing weights.
    def initializePropagation(self, params, FLAGS):

        loadFromFile = not FLAGS.generatePropagation

        self.propagation = {}
        spec = self.spec

        location = params['dirList']['data'] + self.module_name + '-' + self.name + '/'

        spec['location'] = location

        if not os.path.exists(location):
            os.makedirs(location)

        if not self.host_param:
            # params['params'][name] = params
            return 0


        # Propagation function of individual figure
        if loadFromFile:

            self.readPropagation(location, self.spec['grain'], [self.spec['grain'], self.spec['grain']], True, self.channel)

        else:
            if spec['mode'] == 'Measurement':
                self.measurePropagation(spec, params, FLAGS)
            else:
                print('Unkown device mode : ', spec['mode'])

            t_io.WriteJson(location + 'spec.json', self.spec)

            print(self.module_name + ': Saving propagation ...')

            self.initializeScopicWeight(spec['channel'])
            self.writePropagation(location, self.spec['grain'], [self.spec['grain'], self.spec['grain']], True, self.channel)

        return 0

    # Prepare necessary data (like neuron connections) for efficient GPU training
    def initializeTensor(self, params, FLAGS, relase_unused=False):

        device = params['device']

        channel = self.channel
        spec = self.spec

        propagation_tensors = {}

        oCountP = spec['oVirtualHeightP'] * spec['oVirtualWidthP']
        iCountP = spec['iVirtualWidthP'] * spec['iVirtualHeightP']
        lCountP = spec['lVirtualHeightP'] * spec['lVirtualWidthP']

        LCCount = spec['LCCount']

        grain_s = self.spec['grain']

        def AttachKeyChannel(key, c):

            if self.one_channel_propagation >= 0:
                return key + '_c' + str(self.one_channel_propagation)

            else:
                return key + '_c' + str(c)

        # Obsoleted.
        def ProcessScopicWeight(key, channel):

            return
            propagation_tensors[key + '_c' + str(channel)] = torch.from_numpy(self.propagation[AttachKeyChannel(key, channel)]).to(device)


        tensor_dict = {}
        for c in range(len(channel)):

            if channel[c] == 0:
                continue

            # ScopicWeight
            ProcessScopicWeight('NLScopicWeight', c)

            for l in range(LCCount):
                ProcessScopicWeight('LC' + str(l) + 'ScopicWeight', c)

            # Propagation
            propagation = self.propagation


            for g in range(len(grain_s)):
                grain = grain_s[g]

                if g == 0:

                    init_NLPropagation = propagation[AttachKeyChannel('NLPropagation', c) + '_g' + str(grain)].copy()
                    threshold = 1e-1 * init_NLPropagation.max()
                    init_NLPropagation[(init_NLPropagation) < threshold] = threshold

                    init_NLScaleFactor = np.mean(init_NLPropagation)  # * (FLAGS.dNLWidthF * FLAGS.dNLHeightF)
                    init_NLPropagation = init_NLPropagation / init_NLScaleFactor

                    # To match various exposures
                    init_NLScaleFactor *= self.fullPowerOutput.mean()

                    init = np.empty((1), dtype=np.float32)
                    init[0] = init_NLScaleFactor

                    key = AttachKeyChannel('NLPropagationScaleFactor', c) + '_g' + str(grain)
                    if not key in tensor_dict:
                        tensor_dict[key] = torch.from_numpy(init).to(device)
                    propagation_tensors['NLPropagationScaleFactor' + '_c' + str(c) + '_g' + str(grain)] = tensor_dict[key]

                    key = AttachKeyChannel('NLPropagation', c) + '_g' + str(grain)
                    if not key in tensor_dict:
                        tensor_dict[key] = torch.from_numpy(init_NLPropagation).to(device)

                    propagation_tensors['NLPropagation' + '_c' + str(c) + '_g' + str(grain)] = tensor_dict[key]


                for l in range(LCCount):

                    key = AttachKeyChannel('LC'+str(l)+'Propagation', c) + '_g' + str(grain)

                    dense_shape = [oCountP * iCountP // (grain * grain * grain * grain), lCountP // (grain * grain)]
                    # dense_shape = propagation[ptr].shape

                    if not key in tensor_dict:
                        coo_matrix = None

                        coo_matrix = propagation[key]
                        print(key)
                        tensor_dict[key] = t_partitioned_sparse.Matrix(key, dense_shape, 1000000, params['device'], coo_matrix)

                    propagation_tensors['LC'+str(l)+'Propagation_c'+ str(c) + '_g' + str(grain)] = tensor_dict[key]


        self.propagationTensors = propagation_tensors

        if relase_unused:
            self.propagation = None


    def measurePropagation(self, spec, params, FLAGS):

        channel = self.channel

        self.propagation = {}

        startTime = time.time()

        print(self.module_name + ': Calculating NL propagation ...')

        # initialize bg
        lc0 = self.LCs[0]
        lc1 = self.LCs[1]
        input_plane = self.inputPlane
        output_plane = self.outputPlane

        bg_max = output_plane.bg.max()

        grain_s = self.spec['grain']

        rg = list(reversed(range(len(grain_s))))

        for g in rg:
            if g > 0:
                continue
            grain = grain_s[g]
            threshhold = spec['threshold'][g]

            print('Measuring grain', grain, 'in', grain_s)

            # apply grain

            g_input_plane = self.ComponentInterface(self.inputPlane, [0, self.inputPlane.hwc()[0]], [0, self.inputPlane.hwc()[1]], grain, grain, self)
            g_lc0 = self.ComponentInterface(lc0, [0, lc0.hwc()[0]], [0, lc0.hwc()[1]], grain, grain, self)
            g_lc1 = self.ComponentInterface(lc1, [0, lc1.hwc()[0]], [0, lc1.hwc()[1]], grain, grain, self)
            g_output_plane = self.ComponentInterface(self.outputPlane, [0, self.outputPlane.hwc()[0]], [0, self.outputPlane.hwc()[1]], grain, grain, self)
            g_NLPropagation_rgb = [None, None, None]

            # tune sensitivity

            sensitivity = 1.0
            i_hwc = g_input_plane.hwc()

            g_lc0.emit(np.ones(g_lc0.hwc()) * channel)
            g_lc1.emit(np.ones(g_lc1.hwc()) * channel)
            i_impulse = np.zeros(g_input_plane.hwc())
            i_impulse[int(i_hwc[0] / 2), int(i_hwc[1] / 2), :] = 1
            g_input_plane.emit(i_impulse * channel)
            g_input_plane.delay()


            while 1:
                o_img = output_plane.sense()
                if o_img.max() + bg_max > 0.95:
                    sensitivity *= 0.5
                    output_plane.tuneSensitivity(sensitivity)
                else:
                    break

            #################### NL Propagation
            while 1:
                NLPropagation_rgb = MeasurePlanePropagation_input(g_input_plane, g_output_plane, [g_lc0, g_lc1], channel)

                for c in range(len(channel)):
                    print('mean and max ', NLPropagation_rgb[c].mean(), NLPropagation_rgb[c].max())

                over_exposure = False
                for c in range(len(channel)):
                    if channel[c] == 0:
                        continue
                    if NLPropagation_rgb[c].max() + bg_max > 0.99:
                        over_exposure = True

                if over_exposure:
                    sensitivity *= 0.5
                    output_plane.tuneSensitivity(sensitivity)
                else:
                    break

            for c in range(len(channel)):
                 if channel[c] == 0:
                      continue
                 key = 'NLPropagation' + '_c' + str(c) + '_g' + str(grain)
                 self.propagation[key] = NLPropagation_rgb[c]
                 print('mean and max of ' + key, self.propagation[key].mean(), self.propagation[key].max())

            self.writePropagation(spec['location'], [grain], [[],[]], False, channel)

            self.readPropagation(spec['location'], [grain], [[], []], False, channel)
            for c in range(len(channel)):
                if channel[c] == 0:
                    continue
                g_NLPropagation_rgb[c] = self.propagation.get('NLPropagation' + '_c' + str(c) + '_g' + str(grain))


            print(self.module_name + ': Calculating LC propagation 0 ...')

            ##################### LC0

            LC0Propagation_rgb = MeasurePlanePropagation_LC0(g_input_plane, g_lc0, g_lc1, g_output_plane, channel, g_NLPropagation_rgb, threshhold)

            for c in range(len(channel)):
                if channel[c] == 0:
                    continue
                self.propagation['LC0Propagation_c' + str(c) +'_g' + str(grain)] = LC0Propagation_rgb[c]
                print(' NNN Z ', LC0Propagation_rgb[c].nnz)

            self.writePropagation(spec['location'], [], [[grain], []], False, channel)

            self.readPropagation(spec['location'], [], [[grain], []], False, channel)

            ############## LC1
            LC1Propagation_rgb = MeasurePlanePropagation_LC0(g_input_plane, g_lc1, g_lc0, g_output_plane, channel, g_NLPropagation_rgb, threshhold)


            for c in range(len(channel)):
                if channel[c] == 0:
                    continue
                self.propagation['LC1Propagation_c' + str(c) +'_g' + str(grain)] = LC1Propagation_rgb[c]
                print(' NNN Z ', LC1Propagation_rgb[c].nnz)

            self.writePropagation(spec['location'], [], [[], [grain]], False, channel)

            self.readPropagation(spec['location'], [], [[], [grain]], False, channel)

        output_plane.tuneSensitivity(1.0)


        self.initializeScopicWeight(channel)
        self.writePropagation(spec['location'], [], [[],[]], True, channel)

        return

    # The gain of individual sensors, obsoleted
    def initializeScopicWeight(self, channel):
        return


    def evaluate(self, input_img, lc0, lc1, channel):

        assert (input_img.shape == self.inputPlane.hwc())
        assert (lc0.shape == self.LCs[0].hwc())
        assert (lc1.shape == self.LCs[1].hwc())

        i_hwc = self.inputPlane.hwc()
        o_hwc = self.outputPlane.hwc()
        l_hwc = self.LCs[0].hwc()

        grain_s = self.spec['grain']
        response_rgb = np.zeros(o_hwc, np.float32)

        for c in range(len(channel)):

            lc0_scopic_weight = self.propagation.get('LC0ScopicWeight_c' + str(c))
            lc1_scopic_weight = self.propagation.get('LC1ScopicWeight_c' + str(c))
            nl_scopic_weight = self.propagation.get('NLScopicWeight_c' + str(c))

            if channel[c] == 0:
                continue

            def ScopicLayer(input_img, scopic_weight, propagation_prefix):

                lc_transmission_grain = []
                for g in range(0, len(grain_s)):

                    grain = grain_s[g]

                    if not input_img is None:
                        lc_propagation = self.propagation.get(propagation_prefix + '_c' + str(c) + '_g' + str(grain))
                        lc = t_utility.DownScale(input_img, grain, grain)
                        lc = np.reshape(lc, ((l_hwc[0] * l_hwc[1]) // (grain * grain), 1))
                        lc_transmission = lc_propagation.dot(lc.flatten())  # grain oh * grain ow * grain ih * grain iw
                    else:
                        if g == 0:
                            lc_propagation = self.propagation.get(propagation_prefix + '_c' + str(c) + '_g' + str(grain))
                            lc_transmission = lc_propagation
                        else:
                            continue


                    lc_transmission = np.reshape(lc_transmission, (-1, i_hwc[0] // grain, i_hwc[1] // grain))
                    lc_transmission = t_utility.UpScale(lc_transmission, [1, grain, grain]) # (grain oh * grain ow, ih, iw)
                    lc_transmission = np.reshape(lc_transmission, (o_hwc[0] // grain, o_hwc[1] // grain, -1))
                    lc_transmission = t_utility.UpScale(lc_transmission, [grain, grain, 1])  # (oh,  ow, ih * iw)

                    lc_transmission = lc_transmission.reshape((o_hwc[0] * o_hwc[1] * i_hwc[0] * i_hwc[1],))   # (oh * ow * ih * iw)
                    lc_transmission = np.multiply(lc_transmission, scopic_weight[g,:])  # (oh * ow, grain ih, grain iw)
                    lc_transmission_grain.append(lc_transmission.reshape((o_hwc[0] * o_hwc[1], i_hwc[0] * i_hwc[1])))

                lc_transmission = np.add.reduce(lc_transmission_grain)
                return lc_transmission


            lc0_transmission = ScopicLayer(lc0[:,:,c], lc0_scopic_weight, 'LC0Propagation')
            lc1_transmission = ScopicLayer(lc1[:,:,c], lc1_scopic_weight, 'LC1Propagation')


            transmission = np.multiply(lc0_transmission, lc1_transmission)

            # NL
            t = input_img[:, :, c]
            #nl_propagation = self.propagation.get('NLPropagation_c' + str(c) + '_g' + str(grain_s[-1]))
            nl_propagation = ScopicLayer(None, nl_scopic_weight, 'NLPropagation')
            nl_transmission = nl_propagation.reshape((o_hwc[0] * o_hwc[1], i_hwc[0] * i_hwc[1]))


            transmission = np.multiply(nl_transmission, transmission)
            #transmission = nl_transmission

            t = transmission @ t.flatten()

            response_rgb[:,:,c] = t.reshape((o_hwc[0], o_hwc[1]))

        return response_rgb



def CalibrateCrop(spec, device, prefix):
    hwc = device.hwc()

    # height
    physical_height = hwc[0]
    virtual_height = spec[prefix + 'VirtualHeightP']

    scalability_height_p = int(physical_height / virtual_height)
    residual = physical_height - (scalability_height_p * virtual_height)
    crop_height_p = [int(residual / 2), int(residual / 2) + scalability_height_p * virtual_height]

    assert (crop_height_p[1] >= crop_height_p[0] and (crop_height_p[1] - crop_height_p[0]) % scalability_height_p == 0)

    # width
    physical_width = hwc[1]
    virtual_width = spec[prefix + 'VirtualWidthP']

    scalability_width_p = int(physical_width / virtual_width)
    residual = physical_width - (scalability_width_p * virtual_width)
    crop_width_p = [int(residual / 2), int(residual / 2) + scalability_width_p * virtual_width]

    assert (crop_width_p[1] >= crop_width_p[0] and (crop_width_p[1] - crop_width_p[0]) % scalability_width_p == 0)

    return crop_height_p, crop_width_p, scalability_height_p, scalability_width_p


def MeasurePlanePropagation_input(inputPlane, outputPlane, LCs, channel):
    print('Processing : ', inputPlane.name())

    backup_bg = outputPlane.bg
    outputPlane.bg = None

    i_hwc = inputPlane.hwc()
    o_hwc = outputPlane.hwc()

    i_hwc = inputPlane.hwc()

    o_hwc = outputPlane.hwc()

    sensed_propagation = []

    h_points = i_hwc[0]
    w_points = i_hwc[1]


    for i in range(len(channel)):
        sensed_propagation.append(np.zeros((o_hwc[0] * o_hwc[1], h_points, w_points), dtype=np.float32))

    for c in range(len(channel)):

        if channel[c] == 0:
            continue

        ch = np.zeros_like(channel)
        ch[c] = 1
        for l in LCs:
            l.emit(np.ones(l.hwc()) * ch)
        inputPlane.emit(np.zeros(inputPlane.hwc()))

        inputPlane.delay()
        new_bg = outputPlane.sense()
        outputPlane.bg = new_bg

        for h in range(0, h_points):
            # for h in range(0, 8):
            print('h ', h, h_points)

            for w in range(0, w_points):

                input_img = np.zeros(i_hwc)
                input_img[h, w, c] = 1.0

                inputPlane.emit(input_img)
                inputPlane.delay()

                output_img = outputPlane.sense()

                sensed_propagation[c][:, h, w] = output_img[:, :, c].flatten()

    propagation = []
    for col in range(len(channel)):
        propagation.append(np.reshape(sensed_propagation[col], (o_hwc[0] * o_hwc[1], i_hwc[0] * i_hwc[1])))

    outputPlane.bg = backup_bg
    return propagation


def SearchLCResponse_bruteforce(response, lc, output, lc_off_bg, channel, epsilon=0):
    hwc = lc.hwc()
    i_img = np.zeros(hwc) * channel


    for h in range(0, hwc[0]):
        for w in range(0, hwc[1]):

            ii = i_img.copy()
            ii[h, w, :] = 1
            ii *= channel
            lc.emit(ii)
            lc.delay()

            lc_on_response = output.sense() * channel
            input_one_response = lc_on_response - lc_off_bg

            response[h, w, :, :, :] = input_one_response

    return response


num_search = 0
num_leaf = 0

def SearchLCAttenuationResponse_quadtree(response, h_bound, w_bound, lc_off_bg, threshold, lc,
                                         output_plane, input_plane, channel):
    global num_search
    global num_leaf

    hwc = lc.hwc()

    if (h_bound[0] + 1 >= h_bound[1] and w_bound[0] + 1 >= w_bound[1]):
        isLeaf = True
    else:
        isLeaf = False

    # lc on
    lc_inpulse = np.zeros(hwc)
    lc_inpulse[h_bound[0]:h_bound[1], w_bound[0]:w_bound[1], :] = 1
    lc_inpulse *= channel

    lc.emit(lc_inpulse)
    lc.delay()

    lc_on_response = output_plane.sense() * channel

    num_search += 1
    if isLeaf:
        num_leaf += 1

    input_one_response = lc_on_response - lc_off_bg

    response_id = []

    if np.max(input_one_response) > threshold:
        if isLeaf == True:
            response[h_bound[0], w_bound[0], :, :, :] = input_one_response
            response_id.append((h_bound[0], w_bound[0]))
        else:
            mid_h = int((h_bound[0] + h_bound[1]) / 2)
            mid_w = int((w_bound[0] + w_bound[1]) / 2)

            response_id += SearchLCAttenuationResponse_quadtree(response, (h_bound[0], mid_h), (w_bound[0], mid_w),
                                                                lc_off_bg, threshold, lc, output_plane,
                                                                input_plane, channel)
            response_id += SearchLCAttenuationResponse_quadtree(response, (h_bound[0], mid_h), (mid_w, w_bound[1]),
                                                                lc_off_bg, threshold, lc, output_plane,
                                                                input_plane, channel)
            response_id += SearchLCAttenuationResponse_quadtree(response, (mid_h, h_bound[1]), (w_bound[0], mid_w),
                                                                lc_off_bg, threshold, lc, output_plane,
                                                                input_plane, channel)
            response_id += SearchLCAttenuationResponse_quadtree(response, (mid_h, h_bound[1]), (mid_w, w_bound[1]),
                                                                lc_off_bg, threshold, lc, output_plane,
                                                                input_plane, channel)

    return response_id



def MeasurePlanePropagation_LC0(inputPlane, lc0, lc1, outputPlane, channel, NLPropagation_rgb, threshold_ratio):

    print('Processing : ', lc0.name())

    brute_force = True
    if threshold_ratio > 0.0:
        brute_force = False


    i_hwc = inputPlane.hwc()
    o_hwc = outputPlane.hwc()
    l_hwc = lc0.hwc()

    sparse_propagation_rgb = [None, None, None]

    for c in range(len(channel)):
        if channel[c] == 0:
            continue

        ch = np.zeros_like(channel)
        ch[c] = 1
        lc1.emit(np.ones(lc1.hwc()) * ch)

        # find the maximum intensity as reference
        NLPropagation = NLPropagation_rgb[c]
        NLPropagation = np.reshape(NLPropagation, (o_hwc[0], o_hwc[1], i_hwc[0], i_hwc[1]))
        max_value = NLPropagation.max()
        points = np.where(NLPropagation == max_value)
        point = (points[2][0], points[3][0])

        i_response = np.zeros(i_hwc)
        i_response[point[0], point[1], c] = 1
        inputPlane.emit(i_response)
        lc0.emit(np.ones(lc0.hwc()) * ch)
        lc0.delay()
        o_img = outputPlane.sense() * ch
        threshold = o_img.max() * threshold_ratio

        data = []
        row = []
        col = []

        new_start = True



        for ih in range(0, i_hwc[0]):


            if not new_start:
                t_io.WriteJson('temp/row.buffer', row)
                t_io.WriteJson('temp/col.buffer', col)
                t_io.WriteJson('temp/data.buffer', data)
                print('Done writing buffer, len is ', len(data), ', can restart from ih = ', ih)

            new_start = False

            # for iw in range(0, 1):
            for iw in range(0, i_hwc[1]):

                input_one_inpulse = np.zeros(inputPlane.hwc())
                input_one_inpulse[ih, iw, c] = 1
                inputPlane.emit(input_one_inpulse)

                lc0.emit(np.ones(lc0.hwc()) * ch)
                lc0.delay()

                one_point_response = outputPlane.sense()[:, :, c]
                ref_one_point_response = NLPropagation[:, :, ih, iw]

                response = np.zeros((l_hwc[0], l_hwc[1], o_hwc[0], o_hwc[1], 3), dtype=np.float32)

                lc0.emit(np.zeros(lc0.hwc()))
                lc0.delay()
                lc_off_bg = outputPlane.sense() * ch


                if brute_force == False:
                    response_id = SearchLCAttenuationResponse_quadtree(response, (0, l_hwc[0]), (0, l_hwc[1]),
                                                                       lc_off_bg,
                                                                       threshold, lc0, outputPlane,
                                                                       inputPlane, ch)
                else:
                     response = SearchLCResponse_bruteforce(response, lc0, outputPlane, lc_off_bg, channel)

                response = response[:, :, :, :, c]

                bcount = len(data)

                for lh in range(l_hwc[0]):
                    for lw in range(l_hwc[1]):
                        for oh in range(o_hwc[0]):
                            for ow in range(o_hwc[1]):
                                if brute_force or response[lh, lw, oh, ow] >= 0.1 * threshold:
                                    base = one_point_response[oh, ow]
                                    if brute_force:
                                        base = max(0.001, base)
                                    elif base <= 0:
                                        continue  # for gt

                                    row_id = (oh * o_hwc[1] + ow) * (i_hwc[0] * i_hwc[1]) + (ih * i_hwc[1] + iw)
                                    col_id = (lh * l_hwc[1] + lw)

                                    row.append(row_id)
                                    col.append(col_id)

                                    d = min(max(response[lh, lw, oh, ow] / base, 0), 1)
                                    data.append(d)


        a = sparse.csr_matrix((data, (row, col)),
                              shape=(o_hwc[0] * o_hwc[1] * i_hwc[0] * i_hwc[1], l_hwc[0] * l_hwc[1]),
                              dtype=np.float32)
        a = sparse.coo_matrix(a)
        sparse_propagation_rgb[c] = a


    return sparse_propagation_rgb



