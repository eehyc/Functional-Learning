from matplotlib import pyplot as plt
import numpy as np
import math
import t_io
import time
import t_component_interface
import sys
import scipy



def DetectResponseBound(i_camera, i_display, stride = 1):

    h, w, c = i_display.hwc()

    img = np.zeros((h, w, c))
    i_display.emit(img)
    i_display.delay()
    bg = t_io.ToHDR(i_camera.sense())
    print('bg ', bg.sum())

    # horizontal
    y = 0
    hx = []
    hy = []
    while y < h:
        img.fill(0)
        img[y: min(y + stride, h), :, :] = 1
        i_display.emit(img)
        i_display.delay()
        cp = t_io.ToHDR(i_camera.sense())
        output = cp - bg
        print('H %i - %i : %f %f %f' % (y, min(y + stride, h), cp.sum(), output.sum(), output.max()))
        hx.append(y)
        hy.append(output.max())
        y += stride

    plt.plot(hx, hy, marker='.', label='height')



    # vertical
    x = 0
    wx = []
    wy = []
    while x < w:
        img.fill(0)
        img[:, x: min(x + stride, w), :] = 1
        i_display.emit(img)
        i_display.delay()
        cp = t_io.ToHDR(i_camera.sense())
        output = cp - bg
        print('W %i - %i : %f %f %f' % (x,  min(x + stride, w), cp.sum(), output.sum(), output.max()))
        wx.append(x)
        wy.append(output.max())
        x += stride

    plt.plot(wx, wy, marker='o', label='width')
    plt.show()
    plt.pause(1)
    input('done detection')

def CaptureImages(bg, output_plane, input_plane_s, x_array_s, channel_s):

    n_input = len(input_plane_s)

    hwc = [p.hwc() for p in input_plane_s]

    y_img_array = []

    for i in range(len(x_array_s[0])):

        if i % 100 == 0:
            print(i, len(x_array_s[0]))

        for n in range(n_input):

            if x_array_s[n].dtype == np.uint8:
                frame = np.empty(shape=hwc[n], dtype=np.uint8)
                frame.fill(x_array_s[n][i])
            else:
                frame = np.ones(hwc[n]) * x_array_s[n][i]
                # frame[int(size[0]/2):-1,:,:] = 0
                # frame[:,int(size[1]/2):-1,:] = 0
            frame *= np.array(channel_s).astype(frame.dtype)
            input_plane_s[n].emit(frame)

        input_plane_s[0].delay()
        img = output_plane.sense()
        # for i in range(2):
        #     img += output_plane.sense()
        # img /= 3
        img -= bg
        # img = output_plane.sense() - bg
        y_img_array.append(img)

    return y_img_array


def CaptureSequenceOfImages(bg, output_plane, input_plane, step, channel_s):

    if step < 1.0:
        x_array = np.arange(0.0, 1.0 + step, step)
    else:
        x_array = np.arange(0, 256, step).astype(np.uint8)
        x_array[-1] = 255

    y_img_array = CaptureImages(bg, output_plane, [input_plane], [x_array], channel_s)

    if step >= 1.0:
        x_array = x_array.astype(np.float32)
        x_array /= x_array[-1]

    return x_array, y_img_array



# Crop useless pixels from the input plane, output plane, lc0, and lc1.
# Group pixels into neurons.
# This part must be tuned by hand once the hardware is built.
def CalibrateCrop(input_plane, lc0, lc1, output_plane, resize=1):

    stride = 10
    shrink = 0.5 * (1 - resize)

    # h, w, c = input_plane.hwc()
    # black = np.zeros(input_plane.hwc())
    # white = np.ones(input_plane.hwc())
    # dot = np.zeros(input_plane.hwc())
    # half_h = int(0.5 * h)
    # half_w = int(0.5 * w)
    # dot[half_h - 10:half_h+10, half_w - 10 : half_w +10, :] = 1

    # crop calibration, done by hand for now

    # 1. crop input by bounding the peak visible point
    # input_plane.respond(np.zeros(input_plane.hwc()))
    # lc1.respond(np.zeros(lc1.hwc()))
    # lc2.respond(np.zeros(lc2.hwc()))
    # DetectResponseBound(output_plane, input_plane, stride)

    i_crop = [[200, 680], [350, 830], [0, 3]]
    #i_crop = [[200, 600], [350, 700], [0, 3]]

    length = (int(shrink * (i_crop[0][1] - i_crop[0][0])), int(shrink * (i_crop[1][1] - i_crop[1][0])))
    i_crop[0][0] += length[0]
    i_crop[0][1] -= length[0]
    i_crop[1][0] += length[1]
    i_crop[1][1] -= length[1]

    input_plane = t_component_interface.CropInterface(input_plane, i_crop, 0)


    # 2. crop output by showing a circle

    # c_block = np.zeros(input_plane.hwc())
    #
    # c_block[:, :, 1] = 1.0
    # c_block[10:-11, 10:-11, 1] = 0
    # input_plane.respond(c_block)
    # lc1.respond(np.ones(lc1.hwc()))
    # lc2.respond(np.ones(lc2.hwc()))
    # print('here')
    # while 1:
    #     pass


    if not output_plane == None:
        o_crop = [[490, 900], [590, 1150], [0, 3]]
        #o_crop = [[400, 900], [500, 1100], [0, 3]]

        length = (int(shrink * (o_crop[0][1] - o_crop[0][0])), int(shrink * (o_crop[1][1] - o_crop[1][0])))
        o_crop[0][0] += length[0]
        o_crop[0][1] -= length[0]
        o_crop[1][0] += length[1]
        o_crop[1][1] -= length[1]

        output_plane = t_component_interface.CropInterface(output_plane, o_crop, 0)

    # 3. calibrate lc1 and lc2 by searching bounds that effect output
    # input_plane.emit(np.ones(input_plane.hwc()))
    # lc1.emit(np.ones(lc1.hwc()))
    # lc2.emit(np.ones(lc2.hwc()))
    # DetectResponseBound(output_plane, lc1, stride)

    # input_plane.emit(np.ones(input_plane.hwc()))
    # lc0.emit(np.ones(lc0.hwc()))
    # lc1.emit(np.ones(lc1.hwc()))
    # DetectResponseBound(output_plane, lc1, stride)

    lc0_crop = [[200, 680], [330, 820], [0, 3]]

    length = (int(shrink * (lc0_crop[0][1] - lc0_crop[0][0])), int(shrink * (lc0_crop[1][1] - lc0_crop[1][0])))
    lc0_crop[0][0] += length[0]
    lc0_crop[0][1] -= length[0]
    lc0_crop[1][0] += length[1]
    lc0_crop[1][1] -= length[1]


    lc0 = t_component_interface.CropInterface(lc0, lc0_crop, 0)



    lc1_crop = [[190, 640], [360, 830], [0, 3]]
    lc1_crop = [[200, 680], [300, 850], [0, 3]]

    length = (int(shrink * (lc1_crop[0][1] - lc1_crop[0][0])), int(shrink * (lc1_crop[1][1] - lc1_crop[1][0])))
    lc1_crop[0][0] += length[0]
    lc1_crop[0][1] -= length[0]
    lc1_crop[1][0] += length[1]
    lc1_crop[1][1] -= length[1]

    lc1 = t_component_interface.CropInterface(lc1, lc1_crop, 0)

    ###################
    # nn = np.ones(input_plane.hwc())
    # input_plane.emit(nn)
    #
    #
    # lc0.emit(np.ones(lc0.hwc()) * [0, 1, 0])
    # lc1.emit(np.ones(lc1.hwc()) * [0, 1, 0])
    # lc1.delay()
    #
    # img = output_plane.sense()
    # print(img.shape)
    # img /= img.max()
    # img = cv2.resize(img, (320, 320))
    # t_io.ImShow(img, 'name', 10)
    # while (1):
    #     pass
    #
    #
    # i = 0
    # while (1):
    #
    #     nn = np.zeros(input_plane.hwc())
    #     if i == 32:
    #         i = 0
    #     else:
    #         nn[i,i,1] = 1
    #         i = i + 1
    #
    #     input_plane.emit(nn)
    #     input_plane.delay()
    #
    #     img = output_plane.sense()
    #     print(img.shape)
    #     img /= img.max()
    #     img = cv2.resize(img, (32,32))
    #     t_io.ImShow(img, 'name', 5)
    #     pass

    return input_plane, lc0, lc1, output_plane

def TuneDisplayGamma(camera, displays, display_name, plot = True):

    size = displays.displaySize(display_name)
    frame = np.zeros(shape=(size[1], size[0], 3), dtype=np.uint8)

    displays.switchTexture(display_name, frame)
    displays.refreshAll()

    bg = camera.snapImage()
    bg = t_io.ToHDR(bg)
    print('sum bg ' ,np.sum(bg))
    x_array, y_img_array = CaptureSequenceOfImages(bg, camera, displays, display_name, 0.01)

    # gamma offset
    data_dict = {}
    delta_rgb = []
    for c in range(3):

        y_array = []
        for i in range(len(y_img_array)):
            a = y_img_array[i][:, :, c]- bg[:, :, c]
            y_array.append(a.mean())
            #y_array.append(y_img_array[i][:,:,c].mean())
        if np.abs(y_array[-1] - y_array[0]) <= 1e-3:
            print('Invalid results.')
            print(y_array)
            exit(0)
        print(np.array(y_array))

        y_array = np.array(y_array)

        slop = y_array[-1] - y_array[0]
        gt_y_array = x_array * slop + y_array[0]

        delta = []
        for i in range(1, len(x_array) - 1):
            if y_array[i] - y_array[0] > 0:
                delta.append(math.log((y_array[i] - y_array[0]) / (gt_y_array[i] - y_array[0]), x_array[i]))

        delta_rgb.append(np.mean(delta))

        if plot:

            plt.plot([x_array[0], x_array[-1]], [y_array[0], y_array[-1]], linestyle='--')
            plt.plot(x_array, y_array, marker='.')
            plt.show()
            plt.pause(1)
            x_list = x_array.tolist()
            y_list = y_array.tolist()
            data_dict[str(c)] = {}
            data_dict[str(c)]['x_list'] = x_list
            data_dict[str(c)]['y_list'] = y_list


        continue

        # maximum_img = y_img_array[-1]
        #
        # gt_y_img_array = []
        # for i in range(len(x_array )):
        #     gt_y_img_array.append(x_array[i] * maximum_img)
        #
        #
        # delta = []
        # delta_img = []
        # for i in range(1, len(x_array) - 1):
        #
        #
        #     y = np.reshape(y_img_array[i], (-1))
        #     gt_y = np.reshape(gt_y_img_array[i], (-1))
        #
        #     count = 0
        #     sum = 0
        #     for j in range(gt_y.shape[0]):
        #         if gt_y[j] > 0 and y[j] > 0:
        #             t = math.log(y[j] / gt_y[j], x_array[i])
        #
        #             sum += t
        #             count += 1
        #
        #     delta_img.append(sum / count)
        #     delta_rgb.append(np.mean(delta_img))

    plt.pause(1000)
    print('delta_rgb', delta_rgb)
    displays.offsetGamma(display_name, delta_rgb)

def CalibrateSpectrumCorrelation(bg, output_plane, input_plane, displays):

    size = input_plane.hwc()
    frame = np.zeros(shape=(size), dtype=np.uint8)

    input_plane.emit(frame)
    input_plane.delay()

    spectrum_correlation = []
    for c in range(3):
        channels = [0, 0, 0]
        channels[c] = 1
        x_array, y_img_array = CaptureSequenceOfImages(bg, output_plane, input_plane, 255, channels)

        spectrum_mean = np.mean(y_img_array[-1], axis=(0,1))
        spectrum_mean /= spectrum_mean[c]

        spectrum_correlation.append(spectrum_mean.tolist())

    print('spectrum correlation : ', spectrum_correlation)
    displays.spectrum_correlation = spectrum_correlation

def PlotLinearlity(bg, output_plane, input_plane, wait=10, channels=[1, 1, 1]):

    size = input_plane.hwc()
    frame = np.zeros(shape=size, dtype=np.uint8)

    input_plane.emit(frame)
    input_plane.delay()

    # plot test code
    x_array, y_img_array = CaptureSequenceOfImages(bg, output_plane, input_plane, 1, channels)



    for c in range(3):
        if channels[c] == 0:
            continue
        y_array = []
        for i in range(len(y_img_array)):
            a = y_img_array[i][:, :, c]
            y_array.append(a.mean())
            # y_array.append(y_img_array[i][:,:,c].mean())
        if np.abs(y_array[-1] - y_array[0]) <= 1e-3:
            print('Invalid Results for Calibration.')
            exit(0)
        y_array = np.array(y_array)


        plt.plot([x_array[0], x_array[-1]], [y_array[0], y_array[-1]], linestyle='--')
        plt.plot(x_array, y_array, marker='.')
    plt.show()
    plt.pause(wait)



def TestLinearlity(bg, output_plane, input_plane, lc1, lc2, wait=10, channels=[1, 1, 1]):

    input_frame_one = np.ones(input_plane.hwc()) * channels
    input_frame_random = np.random.random_sample(input_plane.hwc()) * channels
    lc1_frame = np.ones(lc1.hwc()) * channels
    lc1_frame_random = np.random.random_sample(lc1.hwc()) * channels
    lc2_frame = np.ones(lc2.hwc()) * channels
    lc2_frame_random = np.random.random_sample(lc2.hwc()) * channels

    input_plane.emit(input_frame_one * 1)
    lc1.emit(lc1_frame * 1)
    lc2.emit(lc2_frame * 1)
    input_plane.delay()

    full = (output_plane.sense() - bg).mean()

    for i in range(20):
        a = (1-i*0.05)
        input_plane.emit(input_frame_one)
        lc1.emit(lc1_frame)
        lc2.emit(lc2_frame)
        lc1.delay()

        light = (output_plane.sense() - bg).mean()
        lc2.emit(lc2_frame * 0.8)
        lc1.emit(lc1_frame * a)
        lc1.delay()
        dark = (output_plane.sense() - bg).mean()

        print(i, 0.8 * a, dark / light)
    exit(0)


    # input_plane.respond(input_frame_one)
    # lc1.respond(lc1_frame * 1)
    # lc2.respond(lc2_frame * 1)
    # input_plane.delay()
    #
    # full_temp = (output_plane.sense() - bg).mean()

    # for i in range(10):
    #
    #     a = 1
    #     b = 0
    #     c = 0.1 * (i + 1)
    #
    #     input_plane.respond(input_frame_random)
    #     lc1.respond(lc1_frame * 1)
    #     lc2.respond(lc2_frame * c)
    #     input_plane.delay()
    #
    #     full_temp = (output_plane.sense() - bg).mean()
    #
    #
    #     input_plane.respond(input_frame_random * a)
    #     lc1.respond(lc1_frame * b)
    #     lc2.respond(lc2_frame * c)
    #     lc2.delay()
    #     img = (output_plane.sense() - bg).mean()
    #     attanuation_1 = img / full_temp
    #     print(attanuation_1)
    # exit(0)
    #print(a, b, c, img)
    #

    a = 1
    b = 0
    c = 1

    input_plane.emit(input_frame_one)
    lc1.emit(lc1_frame * 1)
    lc2.emit(lc2_frame * c)
    input_plane.delay()

    full_temp = (output_plane.sense() - bg).mean()


    input_plane.emit(input_frame_one * a)
    lc1.emit(lc1_frame * b)
    lc2.emit(lc2_frame * c)
    lc2.delay()
    img = (output_plane.sense() - bg).mean()
    attanuation_1 = img / full_temp
    print(attanuation_1)

    a = 1
    b = 1
    c = 0
    input_plane.emit(input_frame_one * a)
    lc1.emit(lc1_frame * b)
    lc2.emit(lc2_frame * c)
    lc2.delay()
    img = (output_plane.sense() - bg).mean()
    attanuation_2 = img / full_temp
    #print(a, b, c, img)
    print(attanuation_1, attanuation_2)


    mm = 0
    dd = 0
    ss = 0
    for i in range(50):
        a = np.random.rand()
        a = 1
        # input_plane.respond(input_frame * a)
        # lc1.respond(lc1_frame)
        # lc2.respond(lc2_frame)
        # lc2.delay()
        #
        # full = (output_plane.sense() - bg).mean()

        b = np.random.rand()
        c = np.random.rand()

        expected = a * full * (attanuation_1 +  (1 - attanuation_1) * b) * (attanuation_2 + (1 - attanuation_2) * c)
        input_plane.emit(input_frame_one * a)
        lc1.emit(lc1_frame * b)
        lc2.emit(lc2_frame * c)
        lc1.delay()
        img = (output_plane.sense() - bg).mean()
        mm += (img - expected) / expected
        dd += (img - expected)
        ss += expected
        print(b, c, expected, img, (img - expected) / expected)
    print(mm / 50, dd, ss, dd / ss)
    exit(0)

    a = 0.5
    b = 0.3
    c = 0.3
    input_plane.emit(input_frame * a)
    lc1.emit(lc1_frame * b)
    lc2.emit(lc2_frame * c)
    lc2.delay()
    img = (output_plane.sense() - bg).mean()
    attanuation_3 = img / standard
    print(a, b, c, img)
    print('att ', attanuation_1, attanuation_2, attanuation_3, attanuation_1 * attanuation_2)

    exit(0)


    step = 0.01
    t_array = np.arange(0.0, 1.0 + step, step)

    x_array_0 = []
    x_array_1 = []
    for i in range(len(t_array)):
        t = t_array[i]
        a = np.random.rand()
        a = 0.5
        b = t / a
        x_array_0.append(a)
        x_array_1.append(b)

    x_array_0 = np.array(x_array_0)
    x_array_1 = np.array(x_array_1)



    for c in range(3):
        if c not in channels:
            continue

        y_img_array = CaptureImages(bg, output_plane, [lc1, lc2], [x_array_0, x_array_1], channels)

        y_array = []
        for i in range(len(y_img_array)):
            a = y_img_array[i][:, :, c]
            y_array.append(a.mean())
            # y_array.append(y_img_array[i][:,:,c].mean())
        if np.abs(y_array[-1] - y_array[0]) <= 1e-3:
            print('Invalid Results for Calibration.')
            exit(0)
        y_array = np.array(y_array)


        plt.plot([t_array[0], t_array[-1]], [y_array[0], y_array[-1]], linestyle='--')
        plt.plot(t_array, y_array, marker='.')

    plt.show()
    plt.pause(wait)

def MeasureLCResponse(bg, input_plane, lc1, lc2, output_plane, displays, shrink_factor_s, channel_s):

    measure_size = np.array([256/ shrink_factor_s[0], 256 / shrink_factor_s[1], 256 / shrink_factor_s[2]]).astype(np.int32)
    measure_size += 1


    table_rgb = []
    #table_rgb = t_io.ReadJson('buffer.MeasureLCResponse')

    for c in range(3):

        c_table = []

        if channel_s[c] != 0:

            for input_idx in range(measure_size[0]):
                if input_idx == 0:
                    input_idx = 1
                print('input ', input_idx)
                input_frame = np.zeros(shape = input_plane.hwc(), dtype=np.uint8)
                input_frame[:,:,c] = min(255, input_idx * shrink_factor_s[0])
                input_plane.emit(input_frame)

                input_table = []

                for lc1_idx in range(measure_size[1]):
                    print('lc1 ', lc1_idx)
                    lc1_frame = np.zeros(shape=lc1.hwc(), dtype=np.uint8)
                    lc1_frame[:, :, c] = min(255, lc1_idx * shrink_factor_s[1])
                    lc1.emit(lc1_frame)

                    x_array = np.arange(0, 257, shrink_factor_s[2]).astype(np.uint8)
                    x_array[-1] = 255

                    y_img_array = CaptureImages(bg, output_plane, [lc2], [x_array], channel_s)
                    y_array = []
                    for i in range(len(y_img_array)):
                        a = y_img_array[i][:, :, c]
                        y_array.append(a.mean())

                    input_table.append(y_array)

                c_table.append(input_table)

        table_rgb.append(c_table)

    #t_io.WriteJson('buffer.MeasureLCResponse', table_rgb)

    # use LC_response instead of lut
    displays.LC_response = table_rgb
    # for d in displays.windows:
    #     displays.windows[d].lut = None


def CalibrateDisplayLinearlity(bg, output_plane, input_plane, displays, display_name, channel_s):

    gt_y_rgb = [None, None, None]
    y_rgb = [None, None, None]

    for c in range(3):
        if channel_s[c] == 0:
            continue
        x_array, y_img_array = CaptureSequenceOfImages(bg, output_plane, input_plane, 1, channel_s)

        y_array = []
        for i in range(len(y_img_array)):
            a = y_img_array[i][:, :, c]
            y_array.append(a.mean())
        if np.abs(y_array[-1] - y_array[0]) <= 1e-3:
            print('Invalid results.')
            print(y_array)
            exit(0)

        y_array = np.array(y_array)
        y_rgb[c] = (y_array)

        slope = y_array[-1] - y_array[0]
        gt_y_array = x_array * slope + y_array[0]
        gt_y_rgb[c] = (gt_y_array)

    displays.calibrateLinearLUT(display_name, 256, y_rgb, gt_y_rgb, channel_s)

def CalibrateAttenuationLinearlity(bg, output_plane, input_plane, displays, display_name, channel_s):

    gt_y_rgb = [None, None, None]
    y_rgb = [None, None, None]

    for c in range(3):
        if channel_s[c] == 0:
            continue

        i_frame = np.zeros(input_plane.hwc())
        i_frame[:,:,c] = 1
        input_plane.emit(i_frame)
        input_plane.delay()
        full = output_plane.sense()
        for i in range(2):
            full += output_plane.sense()
        full /= 3
        full = (full - bg).mean()

        x_array, y_img_array = CaptureSequenceOfImages(bg, output_plane, input_plane, 1, channel_s)

        y_array = []
        for i in range(len(y_img_array)):
            a = y_img_array[i][:, :, c]
            y_array.append(a.mean())
        if np.abs(y_array[-1] - y_array[0]) <= 1e-3:
            print('Invalid results.')
            print(y_array)
            exit(0)

        y_array = np.array(y_array)
        y_rgb[c] = (y_array)

        # slope = y_array[-1] - y_array[0]
        gt_y_array = full * x_array
        gt_y_rgb[c] = (gt_y_array)

    displays.calibrateLinearLUT(display_name, 256, y_rgb, gt_y_rgb, channel_s)

def SolveLCPRadian(t_x0, t_x1, t_y0, t_y1, t_xy):

    max_x_idx = np.argmax(t_x0)

    bb = t_x0[0]
    max_t = max(np.max(t_x0), np.max(t_y0))
    min_t = min(np.min(t_x1), np.min(t_y1))
    absorption =  1 - (min_t / max_t)

    input_offset = -scipy.arccos(bb / max_t)

    x_radian_s = 1 - ((1 - (t_x0 / max_t)) / absorption)
    x_radian_s = scipy.arccos(x_radian_s)


    y_radian_s = 1 - ((1 - (t_y0 / max_t)) / absorption)
    y_radian_s = scipy.arccos(y_radian_s)

    y_maximum_radian = np.max(y_radian_s)

    x_radian_s_1 = x_radian_s + y_maximum_radian
    estimated_t_x1 = max_t * (1 - absorption * (1 - np.abs(scipy.cos(x_radian_s + y_maximum_radian))))

    step = np.arange(256)
    plt.plot(step, t_x1, marker='.')
    plt.plot(step, estimated_t_x1, marker='o')
    plt.show()
    plt.pause(888)


    exit(0)


    x_radian_s[0:max_x_idx] += input_offset
    x_radian_s[max_x_idx:-1] -= input_offset

    y_radian_s = scipy.arccos(t_y0 / max_t) - input_offset
    x_maximum_radian_idx = np.argmax(x_radian_s)
    y_maximum_radian_idx = np.argmax(y_radian_s)

    l = x_radian_s.shape[0]

    input_vector = np.append(np.append(x_radian_s, y_radian_s), np.array([input_offset, max_t]))
    a = 1
    def func(input_vector):

        x = input_vector[0:l]
        y = input_vector[l:2*l]
        i_off = input_vector[2*l]
        m_t = input_vector[2*l+1]

        f = []
        for i in range(l):
            f.append(m_t * (1 - a * (1 - np.abs(scipy.cos(x[i] + y[i] + i_off))) - t_xy[i]))
        for i in range(l):
            f.append(m_t * (1 - a * (1 - np.abs(scipy.cos(x[i] + i_off))) - t_x0[i]))
        for i in range(l):
            f.append(m_t * (1 - a * (1 - np.abs(scipy.cos(x[i] + y[y_maximum_radian_idx] + i_off))) - t_x1[i]))
        for i in range(l):
            f.append(m_t * (1 - a * (1 - np.abs(scipy.cos(y[i] + i_off))) - t_y0[i]))
        for i in range(l):
            f.append(m_t * (1 - a * (1 - np.abs(scipy.cos(y[i] + x[x_maximum_radian_idx] + i_off))) - t_y1[i]))


        return f

    initial_guess = input_vector
    result = scipy.optimize.root(func, initial_guess, method='lm').x
    x_radian_s = result[0:l]
    y_radian_s = result[l:2 * l]
    input_offset = result[2 * l]
    maximum_t = result[2 * l + 1]

    y_maximum_radian = y_radian_s[y_maximum_radian_idx]
    print(y_maximum_radian)
    print(maximum_t)
    print(input_offset)
    print(x_radian_s[x_maximum_radian_idx])

    x_radian_s_0 = scipy.arccos(t_x0 / maximum_t) - input_offset
    x_radian_s_0 += y_maximum_radian
    print(x_radian_s_0)

    x_radian_s_1 = scipy.arccos(t_x1 / maximum_t) - input_offset
    print(x_radian_s_1)
    exit(0)

    step = np.arange(256)
    plt.plot(step, x_radian_s_0, marker='.')
    plt.plot(step, x_radian_s_1, marker='o')
    plt.show()
    plt.pause(1000)

    exit(0)

    return x_radian_s, y_radian_s, input_offset, maximum_t

    sum = 1.0
    off = 0.1

    x = np.array([0.3, 0.5, 0.8, 0.9, 1.2])
    y = np.array([0.2, 0.5, 0.7, 1.0, 1.1])

    # x = 0.3
    # y = 0.2

    t_xy = sum * np.cos(x + y + off) + np.random.rand(5, ) * 0.01 - 0.005
    t_x = sum * np.cos(x + off) + np.random.rand(5,) *0.01 -0.005
    t_y = sum * np.cos(y + off) + np.random.rand(5,) *0.01 -0.005

    input_vector = np.append(np.append(np.append(x, y), off), sum)

    def func(input_vector):

        x = input_vector[0:5]
        y = input_vector[5:10]
        off = input_vector[10]
        sum = input_vector[11]

        f = []
        for i in range(len(x)):
            f.append(sum * scipy.cos(x[i] + y[i] + off) - t_xy[i])
        for i in range(len(x)):
            f.append(sum * scipy.cos(x[i] + off) - t_x[i])
        for i in range(len(x)):
            f.append(sum * scipy.cos(y[i] + off) - t_y[i])

        return f
        #return [scipy.cos(x + y + off) - t_xy, scipy.cos(x + off) - t_x, scipy.cos(y + off) - t_y]

    # f = lambda x, y :x + y + 2 * 3
    # print(f(4, 5))
    # exit(0)

    #input_vector = [0.3, 0.2, 0]
    initialGuess = input_vector
    #initialGuess[0:-1] = 0
    #initialGuess[0] = 0

    #r = scipy.optimize.fsolve(func, input_vector)
    result = scipy.optimize.root(func, initialGuess, method='lm')
    print(result.x)
    print(result.x[0])
    print(result.x[-2])
    print(result.x[-1])
    exit(0)

def CalibrateLCPRadian_Independent(input_plane, lc1, lc2, output_plane, displays):

    ch = 1

    input_plane.emit(np.zeros(input_plane.hwc()))
    lc1.emit(np.zeros(lc1.hwc()))
    lc2.emit(np.zeros(lc2.hwc()))
    lc2.delay()

    bg = t_io.ToHDR(output_plane.sense())
    for i in range(9):
        bg += t_io.ToHDR(output_plane.sense())
    bg /= 10


    # n_sapmle_points = 10
    #
    # lc1_testpoint = np.array([0, 255]).astype(np.uint8)
    # lc1_testpoint = np.append(lc1_testpoint, np.random.randint(253, size=n_sapmle_points).astype(np.uint8) + 1)
    #
    # lc2_testpoint = np.array([0, 255]).astype(np.uint8)
    # lc2_testpoint = np.append(lc2_testpoint, np.random.randint(253, size=n_sapmle_points).astype(np.uint8) + 1)


    lc1_testpoint = np.arange(256).astype(np.uint8)
    lc2_testpoint = np.arange(256).astype(np.uint8)
    np.random.shuffle(lc2_testpoint)


    for c in range(ch):

        i_img = np.zeros(input_plane.hwc())
        i_img[:, :, c] = 1.0
        input_plane.emit(i_img)
        input_plane.delay()


        # x0_img_array_s = CaptureImages(bg, output_plane, [lc1, lc2], [lc1_testpoint, np.zeros_like(lc1_testpoint)], [c])
        # x1_img_array_s = CaptureImages(bg, output_plane, [lc1, lc2], [lc1_testpoint, np.ones_like(lc1_testpoint) * 255], [c])
        # y0_img_array_s = CaptureImages(bg, output_plane, [lc1, lc2], [np.zeros_like(lc1_testpoint), lc2_testpoint], [c])
        # y1_img_array_s = CaptureImages(bg, output_plane, [lc1, lc2], [np.ones_like(lc1_testpoint) * 255, lc2_testpoint], [c])
        # xy_img_array_s = CaptureImages(bg, output_plane, [lc1, lc2], [lc1_testpoint, lc2_testpoint], [c])
        #
        # def convert(i_img_array_s, c):
        #     i_array = []
        #     for i in range(len(i_img_array_s)):
        #         a = i_img_array_s[i][:, :, c]
        #         i_array.append(a.mean())
        #     i_array = np.array(i_array)
        #
        #     if (i_array.max() - i_array.min()) <= 1e-3:
        #         print('Invalid results x0 : ', i_array)
        #         exit(0)
        #     return i_array
        #
        # x0_array = convert(x0_img_array_s, c)
        # x1_array = convert(x1_img_array_s, c)
        # y0_array = convert(y0_img_array_s, c)
        # y1_array = convert(y1_img_array_s, c)
        # xy_array = convert(xy_img_array_s, c)
        #
        # np.save('x0.npy', x0_array)
        # np.save('x1.npy', x1_array)
        # np.save('y0.npy', y0_array)
        # np.save('y1.npy', y1_array)
        # np.save('xy.npy', xy_array)

        x0_array = np.load('x0.npy')
        x1_array = np.load('x1.npy')
        y0_array = np.load('y0.npy')
        y1_array = np.load('y1.npy')
        xy_array = np.load('xy.npy')

        print('x ', lc1_testpoint)
        print('y ', lc2_testpoint)

        x_radian, y_radian, input_radian_offset, max_t = SolveLCPRadian(x0_array, x1_array, y0_array, y1_array, xy_array)

        print(y_radian)
        for i in range(256):
            if y_radian[i] < 0:
                print(i, y_radian[i], lc2_testpoint[i])

        new_y_radian_s = np.zeros_like(y_radian)
        for i in range(256):
            new_y_radian_s[lc2_testpoint[i]] = y_radian[i]
        y_radian = new_y_radian_s
        print(x_radian)
        print(y_radian)
        print(input_radian_offset)
        plt.plot(lc1_testpoint, x_radian, marker='.')
        plt.plot(lc1_testpoint, y_radian, marker='o')
        plt.show()
        plt.pause(1000)

        exit(0)



        y_array_s = []
        for i in range(len(y_img_array)):
            a = y_img_array_s[i][:, :, c]
            y_array_s.append(a.mean())





    # maximum_value_rgb = [0, 0, 0]
    # radian_offset_rgb = [0, 0, 0]
    # reference_rgb = [0, 0, 0]
    #
    # lc1_y_array_rgb = []
    # lc2_y_array_rgb = []
    #
    # # LC1
    # lc2.respond(np.zeros(lc2.hwc()))
    # lc2.delay()
    # for c in range(ch):






        lc1_y_array_rgb.append(y_array)
        reference_rgb[c] = y_array[0]

        max_idx = np.argmax(y_array)
        y_max = y_array[max_idx]
        print('y_max : ', y_max)
        if (y_max > maximum_value_rgb[c]):
            maximum_value_rgb[c] = y_max
            radian_offset_rgb[c] = -np.arccos(reference_rgb[c] / y_max)
            print('LC1 C:%i idx:%i y_max:%f radian_offset:%f' % (c, max_idx, y_max, radian_offset_rgb[c]))

    # LC2
    # lc1.respond(np.zeros(lc1.hwc()))
    # lc1.delay()
    # for c in range(ch):
    #
    #     i_img = np.zeros(input_plane.hwc())
    #     i_img[:,:,c] = 1.0
    #     input_plane.respond(i_img)
    #     input_plane.delay()
    #
    #     x_array, y_img_array = CaptureSequenceOfImages(bg, output_plane, lc2, 1, c)
    #     y_array = []
    #     for i in range(len(y_img_array)):
    #         a = y_img_array[i][:, :, c]
    #         y_array.append(a.mean())
    #
    #     lc2_y_array_rgb.append(y_array)
    #     max_idx = np.argmax(y_array)
    #     y_max = y_array[max_idx]
    #     print('y_max : ', y_max)
    #     if (y_max > maximum_value_rgb[c]):
    #         maximum_value_rgb[c] = y_max
    #         radian_offset_rgb[c] = -np.arccos(reference_rgb[c] / y_max)
    #         print('LC2 C:%i idx:%i y_max:%f radian_offset:%f' % (c, max_idx, y_max, radian_offset_rgb[c]))


    # LC1
    gt_radians_rgb = []
    radians_rgb = []
    for c in range(ch):

        radians = np.arccos(lc1_y_array_rgb[c] / maximum_value_rgb[c])
        min_idx = np.argmin(radians)
        radians[min_idx:-1] -= radian_offset_rgb[c]
        radians[:min_idx] += radian_offset_rgb[c]

        radians = np.abs(raidans)
        radians_rgb.append(radians)

        maximum_radians = np.max(radians)
        gt_radians = x_array * maximum_radians / x_array.max()
        print(gt_radians)
        gt_radians_rgb.append(gt_radians)
    print(len)
    displays.calibrateLinearLUT(lc1.name(), 256, radians_rgb, gt_radians_rgb, 1)

    exit(0)

def CalibrateLCPAngle_navive(input_plane, lc1, lc2, output_plane, displays):

    ch = 1
    input_plane.emit(np.zeros(input_plane.hwc()))
    lc1.emit(np.zeros(lc1.hwc()))
    lc2.emit(np.zeros(lc2.hwc()))
    lc2.delay()

    bg = t_io.ToHDR(output_plane.sense())
    for i in range(9):
        bg += t_io.ToHDR(output_plane.sense())
    bg /= 10

    input_plane.emit(np.ones(input_plane.hwc()))
    input_plane.delay()

    maximum_value_rgb = [0, 0, 0]
    radian_offset_rgb = [0, 0, 0]
    reference_rgb = [0, 0, 0]

    lc1_y_array_rgb = []
    lc2_y_array_rgb = []

    # LC1
    lc2.emit(np.zeros(lc2.hwc()))
    lc2.delay()
    for c in range(ch):

        i_img = np.zeros(input_plane.hwc())
        i_img[:,:,c] = 1.0
        input_plane.emit(i_img)
        input_plane.delay()

        x_array, y_img_array = CaptureSequenceOfImages(bg, output_plane, lc1, 1, c)
        y_array = []
        for i in range(len(y_img_array)):
            a = y_img_array[i][:, :, c]
            y_array.append(a.mean())

        lc1_y_array_rgb.append(y_array)
        reference_rgb[c] = y_array[0]

        max_idx = np.argmax(y_array)
        y_max = y_array[max_idx]
        print('y_max : ', y_max)
        if (y_max > maximum_value_rgb[c]):
            maximum_value_rgb[c] = y_max
            radian_offset_rgb[c] = -np.arccos(reference_rgb[c] / y_max)
            print('LC1 C:%i idx:%i y_max:%f radian_offset:%f' % (c, max_idx, y_max, radian_offset_rgb[c]))

    # LC2
    # lc1.respond(np.zeros(lc1.hwc()))
    # lc1.delay()
    # for c in range(ch):
    #
    #     i_img = np.zeros(input_plane.hwc())
    #     i_img[:,:,c] = 1.0
    #     input_plane.respond(i_img)
    #     input_plane.delay()
    #
    #     x_array, y_img_array = CaptureSequenceOfImages(bg, output_plane, lc2, 1, c)
    #     y_array = []
    #     for i in range(len(y_img_array)):
    #         a = y_img_array[i][:, :, c]
    #         y_array.append(a.mean())
    #
    #     lc2_y_array_rgb.append(y_array)
    #     max_idx = np.argmax(y_array)
    #     y_max = y_array[max_idx]
    #     print('y_max : ', y_max)
    #     if (y_max > maximum_value_rgb[c]):
    #         maximum_value_rgb[c] = y_max
    #         radian_offset_rgb[c] = -np.arccos(reference_rgb[c] / y_max)
    #         print('LC2 C:%i idx:%i y_max:%f radian_offset:%f' % (c, max_idx, y_max, radian_offset_rgb[c]))


    # LC1
    gt_radians_rgb = []
    radians_rgb = []
    for c in range(ch):

        radians = np.arccos(lc1_y_array_rgb[c] / maximum_value_rgb[c])
        min_idx = np.argmin(radians)
        radians[min_idx:-1] -= radian_offset_rgb[c]
        radians[:min_idx] += radian_offset_rgb[c]

        radians = np.abs(raidans)
        radians_rgb.append(radians)

        maximum_radians = np.max(radians)
        gt_radians = x_array * maximum_radians / x_array.max()
        print(gt_radians)
        gt_radians_rgb.append(gt_radians)
    print(len)
    displays.calibrateLinearLUT(lc1.name(), 256, radians_rgb, gt_radians_rgb, 1)

    exit(0)




# TuneDisplay(None, None, None)
#
# N = 50
# x = np.random.rand(N)
# y = np.random.rand(N)
# colors = np.random.rand(N)
# area = np.pi * (15 * np.random.rand(N))**2  # 0 to 15 point radii
# plt.scatter(x, y, s=area, c=colors, alpha=0.5)
# plt.show()
#
# X = np.linspace(-np.pi, np.pi, 256,endpoint=True)
# C,S = np.cos(X), np.sin(X)
#
# plt.plot(X, C, color="blue", linewidth=2.5, linestyle="-")
# plt.plot(X, S, color="red", linewidth=2.5, linestyle="-")
#
# plt.xlim(X.min()*1.1, X.max()*1.1)
# plt.xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi],
# [r'$-\pi$', r'$-\pi/2$', r'$0$', r'$+\pi/2$', r'$+\pi$'])
#
# plt.ylim(C.min()*1.1,C.max()*1.1)
# plt.yticks([-1, 0, +1],
# [r'$-1$', r'$0$', r'$+1$'])
#
# plt.show()