import torch
from timeit import default_timer as timer
import numpy as np
import os
from fnmatch import fnmatchcase as match
import ast
import logging
import fnmatch
import t_io
import collections



class SparsityFunction(object):

    def __init__(self, begin_step, end_step, initial_sparsity, target_sparsity, pruning_frequency = 1, exponent = 3.):

        self.begin_step = begin_step
        self.end_step = end_step
        self.initial_sparsity = initial_sparsity
        self.target_sparsity = target_sparsity
        self.pruning_frequency = pruning_frequency
        self.exponent = exponent



    def Valid(self, step):
        if step >= self.begin_step and step <= self.end_step:
            return True
        else:
            return False

    def __call__(self, step):

        if step < self.begin_step:
            return self.initial_sparsity

        if step > self.end_step:
            return self.target_sparsity

        n = (self.end_step - self.begin_step) / self.pruning_frequency
        st = self.target_sparsity + (self.initial_sparsity - self.target_sparsity) *\
        pow(1 - (step - self.begin_step) / (n * self.pruning_frequency), self.exponent)

        return st

class Constrain(object):

    def __init__(self, parameter, min, max):
        self.parameter = parameter
        self.min = min
        self.max = max

    def __call__(self):
        self.parameter.clamp_(self.min, self.max)

def Digitize(inputs, maximum=4096):

    return (inputs * maximum).floor() / maximum



def GatherList(module, key):

    class GatherUpdates:
        def __init__(self, key):
            self.contains = []
            self.key = key

        def __call__(self, m):
            if hasattr(m, self.key):
                att = getattr(m, self.key)
                if isinstance(att, list):
                    self.contains.extend(att)
                elif isinstance(att, dict):
                    self.contains.extend(att.values())
                else:
                    self.contains.append(att)
                    # assert 0, 'unknown container'

    gather = GatherUpdates(key)
    module.apply(gather)
    return gather.contains

def GatherDict(module, key):

    class GatherUpdates:
        def __init__(self, key):
            self.contains = {}
            self.key = key

        def __call__(self, m):
            if hasattr(m, self.key):
                att = getattr(m, self.key)
                if isinstance(att, list):
                    assert(0)
                elif isinstance(att, dict):
                    for key, v in att.items():
                        self.contains[key] = v
                else:
                    assert(0)
                    # self.contains.append(att)
                    # assert 0, 'unknown container'

    gather = GatherUpdates(key)
    module.apply(gather)
    return gather.contains

def Assign(module, key, value):

    class AssignUpdates:
        def __init__(self, key, value):
            self.value = value
            self.key = key
            self.count = 0

        def __call__(self, m):
            if hasattr(m, self.key):
                setattr(m, self.key, self.value)
                self.count += 1

    assign = AssignUpdates(key, value)
    module.apply(assign)
    return assign.count

def Call(module, key, value):

    class CallUpdates:
        def __init__(self, key, value):
            self.value = value
            self.key = key
            self.count = 0

        def __call__(self, m):
            if hasattr(m, self.key):
                op = getattr(m, self.key)
                op(value)
                self.count += 1

    assign = CallUpdates(key, value)
    module.apply(assign)
    return assign.count

class Accuracy(torch.nn.Module):

    def __init__(self):
        super().__init__()

    # input = n x c vector, target = n x 1 vector
    def forward(self, inputs, target):

        correct_prediction = torch.eq(torch.argmax(inputs, -1), target)
        correct_prediction = correct_prediction.to(torch.float)
        correctMean = torch.mean(correct_prediction)
        return correctMean

class Accuracy(torch.nn.Module):

    def __init__(self):
        super().__init__()

    # input = n x c vector, target = n x 1 vector
    def forward(self, inputs, target):

        correct_prediction = torch.eq(torch.argmax(inputs, -1), target)
        correct_prediction = correct_prediction.to(torch.float)
        correctMean = torch.mean(correct_prediction)
        return correctMean

# Calculates error (NP)
def Error(logits, gt, errorFunc):

    # L2 loss
    if errorFunc == "L2":
        l2Err = (np.linalg.norm(logits - gt) ** 2) / 2
        l2ErrMean = np.mean(l2Err)
        return l2ErrMean

    # LMLS loss
    elif errorFunc == "LMLS":
        r = logits - gt
        lmls = np.mean(np.log(1 + (0.5 * np.mean(np.square(r), reduction_indices=3))))
        return lmls

    # RelMse loss
    elif errorFunc == "RELMSE":
        num = np.square(logits - gt)
        denom = np.square(gt) + 1.0e-2
        relMse = num / denom
        relMseMean = 0.5 * np.mean(relMse)
        return relMseMean

    # L1 loss
    elif errorFunc == "L1":
        l1Err = np.abs(logits - gt)
        l1ErrMean = np.mean(l1Err)
        return l1ErrMean

    # MAPE loss (relative L1)
    elif errorFunc == "MAPE":
        l1Err = np.abs(logits - gt)
        l1Err = np.divide(l1Err, np.abs(gt) + 1.0e-2)
        l1ErrMean = np.mean(l1Err)
        return l1ErrMean

    # SSIM loss
    elif errorFunc == "SSIM":
        return SSIM(logits, gt)


    elif errorFunc == "PredictAccuracy":
        correct_prediction = np.equal(np.argmax(logits, -1), np.argmax(gt, -1))
        correct_prediction = np.cast(correct_prediction, tf.float32)
        correctMean = np.mean(correct_prediction)
        return correctMean

    elif errorFunc == "PSNR":
        mse = ((logits - gt) ** 2).mean()
        PIXEL_MAX = max(np.max(logits), np.max(gt), 1.0)
        psnr = 20 * np.log10(PIXEL_MAX / np.sqrt(mse))
        return psnr

    elif errorFunc == "Zero":

        l1Err = np.abs(logits - logits)
        l1ErrMean = np.mean(l1Err)
        return l1ErrMean

    else:
        print('Error: Unrecognized error function')


class LearningRateDecay:

    def __init__(self, decay_rate, learning_rate):

        self.patience = 0
        self.pre_loss = float("inf")
        self.decay_rate = decay_rate
        self.learning_rate = learning_rate
        self.decay_patience = 10

        self.assign_op = tf.assign(self.learning_rate, self.learning_rate * self.decay_rate)

    def Update(self, loss, sess):

        if self.pre_loss < loss:
            self.patience += 1

        if self.patience >= self.decay_patience:

            sess.run(self.assign_op)
            self.patience = 0

        self.pre_loss = loss

class RegularizationDecay:

    def __init__(self, decay_rate, regular_ratio, reference_penalty):

        self.patience = 0
        self.decay_rate = decay_rate
        self.regular_ratio = regular_ratio
        self.reference_penalty = reference_penalty
        self.decay_patience = 10

        self.assign_op = tf.assign(self.regular_ratio, self.regular_ratio * self.decay_rate)

    def Update(self, penalty, sess):

        if penalty < self.reference_penalty:
            self.patience += 1

        if self.patience >= self.decay_patience:

            sess.run(self.assign_op)
            self.patience = 0


# Update error to start at startIter
def UpdateError(dir, startIter, numItersForEval):

    fp = open(dir + "Error.txt", 'r')
    lines = fp.readlines()
    fp.close()
    index = 0
    fp = open(dir + "Error.txt", 'w')
    for l in lines:
        if (index == startIter):
            break
    fp.write(l)
    index += numItersForEval
    fp.close()

# Filter Variables
def DictFailFilter(a_dict, key_words):

    b_dict = {}

    for name, par in a_dict.items():
        insert = True
        for key in key_words:
            if match(name, key):
                insert = False
                break
        if insert == True:
            b_dict[name] = par

    return b_dict


# Initialize dictionary of errors to zero
def InitializeErrors(errorList):

    errors = {}
    for i in errorList:
        errors[i] = 0.0

    return errors

def AutoEncodeLoader(loader, shuffle, batch_size, worker):

    s_inputs = []
    with torch.no_grad():
        for idx, (inputs, targets) in enumerate(loader):
            s_inputs.append(inputs.cpu())

    inputs = torch.cat(s_inputs, axis=0)

    dataset = torch.utils.data.TensorDataset(inputs, inputs)
    o_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                                                 num_workers=worker)

    return o_loader

def ForwardLoader(loader, model, shuffle, batch_size, worker):

    model.eval()
    device = next(model.parameters()).device

    s_outputs = []
    s_targets = []
    with torch.no_grad():
        for idx, (inputs, targets) in enumerate(loader):
            out = model(inputs.to(device)).cpu()
            s_outputs.append(out)
            s_targets.append(targets)

    outputs = torch.cat(s_outputs, axis=0)
    targets = torch.cat(s_targets, axis=0)

    dataset = torch.utils.data.TensorDataset(outputs, targets)
    o_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                                                 num_workers=worker)
    return o_loader



def SenseLoader(loader, model, shuffle, batch_size, worker):

    model.eval()
    device = next(model.parameters()).device

    s_outputs = []
    s_targets = []
    with torch.no_grad():
        for idx, (inputs, targets) in enumerate(loader):
            out = model.sense(t_io.ImgConvertTorchAndNumpy(inputs))
            out = t_io.ImgConvertTorchAndNumpy(out).to(device)
            s_outputs.append(out)
            s_targets.append(targets)

    outputs = torch.cat(s_outputs, axis=0)
    targets = torch.cat(s_targets, axis=0)

    dataset = torch.utils.data.TensorDataset(outputs, targets)
    o_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                                                 num_workers=worker)
    return o_loader

def EvaluateErrors(loader, model, loss, num_batch=-1):

    device = next(model.parameters()).device
    if num_batch < 0:
        num_batch = len(loader)

    with torch.no_grad():

        model.eval()

        for batch_idx, (inputs, targets) in enumerate(loader):

            if batch_idx >= num_batch:
                break

            N = inputs.size()[0]

            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)

            error = loss(outputs, targets)

            if batch_idx == 0:
                summary = N * error
                sum_N = N
            else:
                summary += N * error
                sum_N += N

        return summary / sum_N

class Dummy:

    def __enter__(self):
        return

    def __exit__(self, type, value, trace):
        return

# convert shape to bhwc
def StandardizeShape(images):

    origin_shape = images.shape

    if len(origin_shape) == 4:
        return images

    if len(origin_shape) == 2:
        return np.expand_dims(np.expand_dims(images, axis=0), axis=-1)

    if len(origin_shape) == 3:
        if origin_shape[2] <= 3:
            return np.expand_dims(images, axis=0)
        else:
            return np.expand_dims(images, axis=-1)

    assert(False)


def mergeParameters(obj, prefix, for_range, stack_dim):


    if type(obj) is dict or type(obj) is collections.OrderedDict:
        merged = []
        for c in for_range:
            key = prefix % (c)
            if key in obj:
                merged.append(obj[key])

        merged = torch.stack(merged, dim=stack_dim)

        return merged
    else:
        merged = []
        for c in for_range:
            key = prefix % (c)
            if hasattr(obj, key):
                merged.append(getattr(obj, key))

        merged = torch.stack(merged, dim=stack_dim)

        return merged


def DownScale(img, grain_s):

    if np.max(grain_s) == 1:
        return img

    if isinstance(img, np.ndarray):

        assert len(grain_s) == img.ndim, "dimension must match"

        sp = img.shape

        reshape_sp = []
        shrink_axis = []
        for i in range(len(sp)):
            reshape_sp.append(sp[i] // grain_s[i])
            reshape_sp.append(grain_s[i])

            shrink_axis.append(2 * i + 1)

        reshape_sp[0] = -1

        reshaped_tensor = np.reshape(img, reshape_sp)
        shrinked_tensor = reshaped_tensor.mean(tuple(shrink_axis))

        return shrinked_tensor

    else:

        sp = img.shape

        assert len(grain_s) == len(sp), "dimension must match"

        reshape_sp = []
        shrink_axis = []
        for i in range(len(sp)):
            reshape_sp.append(sp[i] // grain_s[i])
            reshape_sp.append(grain_s[i])

            shrink_axis.append(2 * i + 1)

        reshape_sp[0] = -1

        reshaped_tensor = img.view(reshape_sp)
        shrinked_tensor = torch.mean(reshaped_tensor, dim=shrink_axis)

        return shrinked_tensor




def DownScaleMax(img, grain_s):

    if np.max(grain_s) == 1:
        return img

    if isinstance(img, np.ndarray):

        assert len(grain_s) == img.ndim, "dimension must match"

        sp = img.shape

        reshape_sp = []
        shrink_axis = []
        for i in range(len(sp)):
            reshape_sp.append(sp[i] // grain_s[i])
            reshape_sp.append(grain_s[i])

            shrink_axis.append(2 * i + 1)

        reshape_sp[0] = -1

        reshaped_tensor = np.reshape(img, reshape_sp)
        shrinked_tensor = reshaped_tensor.max(tuple(shrink_axis))

        return shrinked_tensor

    else:

        sp = img.shape

        assert len(grain_s) == len(sp), "dimension must match"

        reshape_sp = []
        shrink_axis = []
        for i in range(len(sp)):
            reshape_sp.append(sp[i] // grain_s[i])
            reshape_sp.append(grain_s[i])

            shrink_axis.append(2 * i + 1)

        reshape_sp[0] = -1

        reshaped_tensor = img.view(reshape_sp)
        shrinked_tensor = torch.max(reshaped_tensor, dim=shrink_axis)

        return shrinked_tensor

def UpScale(img, grain_s):

    if np.max(grain_s) == 1:
        return img

    if isinstance(img, np.ndarray):

        repeats = grain_s
        tensor = img

        assert len(repeats) == tensor.ndim, "dimension must match"

        repeated = tensor
        for axis, repeat in enumerate(repeats):
            repeated = np.repeat(repeated, repeat, axis=axis)



        return repeated

    else:

        def up_two_dims(tensor, scale_factor, dim):

            ori_size = list(tensor.size())
            target_size = ori_size.copy()

            for d, scale in enumerate(scale_factor):
                target_size[d + dim] *= scale

            scale_dim = len(scale_factor)
            if dim + scale_dim < len(ori_size):
                after_sum = np.prod(ori_size[dim + scale_dim:])
                transform_size = [1,-1] + ori_size[dim : dim + scale_dim] + [after_sum]
                tensor = tensor.view(transform_size)
                tensor = torch.nn.functional.interpolate(tensor, scale_factor=scale_factor+[1])
            else:
                transform_size = [1, -1] + ori_size[dim:]
                tensor = tensor.view(transform_size)
                tensor = torch.nn.functional.interpolate(tensor, scale_factor=scale_factor)


            return tensor.view(target_size)


        repeats = grain_s
        tensor = img

        assert len(repeats) == len(tensor.size()), "dimension must match"

        repeated = tensor

        callout_record = []
        index = len(grain_s)
        while index >= 0:
            ptr = max(index - 2, 0)
            scale = grain_s[max(index - 2, 0):index]
            if len(scale) > 0 and max(scale) > 1:
                callout_record.append((scale, ptr))
            index -= 2

        for (scale, ptr) in reversed(callout_record):
            repeated = up_two_dims(repeated, scale, ptr)

        return repeated


def One_hot(target, num_class):

    if isinstance(target, np.ndarray):
        res = np.eye(num_class)[np.array(target).reshape(-1)]
        return res.reshape(list(target.shape) + [num_class])
    else:
        return torch.zeros(target.size(), num_class).scatter_(1, target, 1)
        # return tf.one_hot(target, num_class)

def SqueezeChannel(img, channel):

    if isinstance(img, np.ndarray):
        channel = np.array(channel)
        to_take = np.argwhere(channel > 0).flatten()
        t_img = np.take(img, to_take, axis=-1)
        return t_img
    else:
        assert 0, 'No implementation.'

def ExpandChannel(img, channel):
    if isinstance(img, np.ndarray):
        channel = np.array(channel)
        to_insert = np.argwhere(channel > 0).flatten()

        t_shape = list(img.shape)
        t_shape[-1] = len(channel)
        t_img = np.zeros(t_shape, dtype=img.dtype)

        for ptr in range(len(to_insert)):
            t_img[...,to_insert[ptr]] = img[...,ptr]

        return t_img
    else:
        assert 0, 'No implementation.'


import inspect
import ctypes

def _async_raise(tid, exctype):
    """raises the exception, performs cleanup if needed"""
    tid = ctypes.c_long(tid)
    if not inspect.isclass(exctype):
        exctype = type(exctype)
    res = ctypes.pythonapi.PyThreadState_SetAsyncExc(tid, ctypes.py_object(exctype))
    if res == 0:
        raise ValueError("invalid thread id")
    elif res != 1:
        ctypes.pythonapi.PyThreadState_SetAsyncExc(tid, None)
        raise SystemError("PyThreadState_SetAsyncExc failed")

def stop_thread(thread):
    _async_raise(thread.ident, SystemExit)

def UnionShuffledCopies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]