import numpy as np
import t_io

class ImageComponentInterface:

    def __init__(self):
        self.upstream = None
        pass

    def delay(self):
        if self.upstream is None:
            assert 0, 'Undefined Function'
        else:
            return self.upstream.delay()

    def name(self):
        if self.upstream is None:
            return 'UnknownDevice'
        else:
            return self.upstream.name()

    def hwc(self):
        if self.upstream is None:
            assert 0, 'Undefined Function'
        else:
            return self.upstream.hwc()

    def emit(self, img):
        if self.upstream is None:
            assert 0, 'Undefined Function'
        else:
            return self.upstream.respond(img)

    def tuneSensitivity(self, scale):
        if self.upstream is None:
            assert 0, 'Undefined Function.'
        else:
            return self.upstream.tuneSensitivity(scale)

    def sense(self):
        if self.upstream is None:
            assert 0, 'Undefined Function'
        else:
            return self.upstream.sense()

class DummyInterface(ImageComponentInterface):

    def __init__(self, h, w, c):
        self.upstream = None
        self.h = h
        self.w = w
        self.c = c


    def hwc(self):
        return (self.h, self.w, self.c)

class CropInterface(ImageComponentInterface):

    # crop should be a tuple containing crop bounds
    def __init__(self, upstream, crop, pad_value = 0):

        self.bg = None
        self.upstream = upstream
        self.crop = crop

        self.pad_value = pad_value
        h, w, c = upstream.hwc()
        self.pad_size = ((crop[0][0],0 if crop[0][1] == -1 else h - crop[0][1]),
                         (crop[1][0],0 if crop[1][1] == -1 else w - crop[1][1]),
                         (crop[2][0],0 if crop[2][1] == -1 else c - crop[2][1]))

    def name(self):
        return self.upstream.name()

    def sense(self):
        img = t_io.ToHDR(self.upstream.sense())
        img = img[self.crop[0][0] : self.crop[0][1], self.crop[1][0] : self.crop[1][1], self.crop[2][0]: self.crop[2][1]]

        if not self.bg is None:
            img -= self.bg
        return img

    def emit(self, img):
        p_img = np.pad(img, self.pad_size, 'constant', constant_values=self.pad_value)

        self.upstream.emit(p_img)


    def delay(self):
        self.upstream.delay()

    def hwc(self):
        return (self.crop[0][1] - self.crop[0][0], self.crop[1][1] - self.crop[1][0], self.crop[2][1] - self.crop[2][0] )


class LDROffsetInterface(ImageComponentInterface):

    def __init__(self, upstream, offset):
        self.upstream = upstream
        self.offset = offset

    def name(self):
        return self.upstream.name()

    def sense(self):
        img = t_io.ToLDR(self.upstream.sense())
        return img + self.offset

    def emit(self, img):
        o_img = t_io.ToLDR(img) + self.offset
        self.upstream.emit(o_img)


    def delay(self):
        self.upstream.delay()

    def hwc(self):
        return self.upstream.hwc()