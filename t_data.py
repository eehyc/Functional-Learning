from os import listdir
import os, shutil
import os.path
from os.path import isfile, join
from shutil import copyfile
import numpy as np
import t_utility
import math
import sklearn.decomposition
import sklearn.preprocessing
import logging
import fnmatch
import torch
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
import t_io



# Set up the results directory and subfolders
def InitializeResultsFolder(params, FLAGS):

    dirList = {}
    # Make output folder
    if FLAGS.outputDir == '':
        o_dir = FLAGS.projectDir + 'output/'
    else:
        o_dir = FLAGS.outputDir
    outputDir = o_dir + FLAGS.experimentName + "/"

    if not os.path.exists(outputDir):
        os.makedirs(outputDir)
    dirList['output'] = outputDir

    # Make checkpoint folder
    logDir = outputDir + "Logs/"
    if not os.path.exists(logDir):
        os.makedirs(logDir)
    dirList['log'] = logDir

    # Make checkpoint folder
    checkpointDir = outputDir + "Checkpoints/"
    if not os.path.exists(checkpointDir):
        os.makedirs(checkpointDir)
    dirList['checkpoint'] = checkpointDir

    if FLAGS.checkpointStage.endswith('\\\\') or FLAGS.checkpointStage.endswith('/'):
        if not os.path.exists(checkpointDir + FLAGS.checkpointStage):
            os.makedirs(checkpointDir + FLAGS.checkpointStage)

    # Make images folder
    imgDir = outputDir + "Images/"
    if not os.path.exists(imgDir):
        os.makedirs(imgDir)
    dirList['image'] = imgDir

    # Make data folder
    dataDir = outputDir + "Data/"
    if not os.path.exists(dataDir):
        os.makedirs(dataDir)
    dirList['data'] = dataDir

    # Make training images folder
    trainImgDir = imgDir + "Training/"
    if not os.path.exists(trainImgDir):
        os.makedirs(trainImgDir)
    dirList['trainImg'] = trainImgDir

    summaryDir = outputDir + "Summary/" + FLAGS.summaryStage
    if not os.path.exists(summaryDir):
        os.makedirs(summaryDir)
    elif not FLAGS.restore:
        # os.system("del -rf %s*" % summaryDir)
        shutil.rmtree(summaryDir)

    dirList['summary'] = summaryDir

    # Make validation images folder
    validationImgDir = imgDir + "Validation/"
    if not os.path.exists(validationImgDir):
        os.makedirs(validationImgDir)
    dirList['validImg'] = validationImgDir

    # Make test images folder
    testImgDir = imgDir + "Test/"
    if not os.path.exists(testImgDir):
        os.makedirs(testImgDir)
    dirList['testImg'] = testImgDir

    # Make a code folder
    codeDir = outputDir + "Code/"
    if not os.path.exists(codeDir):
        os.makedirs(codeDir)

    # Copy code to folder
    onlyFiles = [f for f in listdir('.') if isfile(join('.', f)) and ('.py' in f)]
    for curFile in onlyFiles:
        copyfile(curFile, codeDir + curFile)

    params['dirList'] = dirList

    return FLAGS, params

def LastCheckpoints(checkpointDir, checkpointName):

    all_files = os.listdir(checkpointDir)

    checkpoint_files = []
    for file in all_files:
        if fnmatch.fnmatch(file, checkpointName + "*.pt"):
            checkpoint_files.append(file)

    checkpoint_files.sort(key=lambda fn: os.path.getmtime(checkpointDir + fn))

    return checkpoint_files



def InitializeStep(params, FLAGS):

    globalStep = 1
    stage = 1

    dir = params['dirList']['checkpoint'] + FLAGS.startCheckpointStage
    name = FLAGS.experimentName + '.s.*.step.'
    checkpoint_files = LastCheckpoints(dir, name)

    if FLAGS.restore:
        if FLAGS.startStep >= 0:

            globalStep = FLAGS.startStep

            found = False
            for file in reversed(checkpoint_files):
                if fnmatch.fnmatch(file, '*.step.'+ str(globalStep) + '*'):
                    found = True
                    stage = int(os.path.basename(file).split('.')[-4])
                    break

            assert found, 'Cannot find check point file : ' + (params['dirList']['checkpoint'] + FLAGS.checkpointStag + FLAGS.experimentName+'.s.*.step.'+str(globalStep)+'.pt')
        elif FLAGS.startStep == -2:

            stage = FLAGS.startStage

            found = False
            for file in reversed(checkpoint_files):
                if fnmatch.fnmatch(file, '*.s.' + str(stage) + '.*'):
                    found = True
                    globalStep = int(os.path.basename(file).split('.')[-2])
                    break

            assert found, 'Cannot find check point file : ' + (dir + FLAGS.experimentName + '.s.' + str(stage) + '.step.*.pt')

        elif FLAGS.startStep == -1:

            def LastStep(checkpointDir, checkpointName):

                latest_file = checkpoint_files[-1]
                step = int(os.path.basename(latest_file).split('.')[-2])
                stage = int(os.path.basename(latest_file).split('.')[-4])

                return stage, step

            stage, globalStep = LastStep(dir, name)

            if globalStep < 0:
                stage, globalStep = LastStep(params['dirList']['checkpoint'] + FLAGS.startCheckpointStage, FLAGS.experimentName)

        else:
            assert 0, 'Unknown flags to restore checkpoint.'

    params['globalStep'] = globalStep
    params['stage'] = stage

    # log writer
    TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now())
    dir = params['dirList']['log'] + TIMESTAMP
    params['dirList']['log'] = dir
    params['logWriter'] = None

    return globalStep

def GetLogWriter(params, FLAGS):

    if params['logWriter'] is None:
        writer = SummaryWriter(params['dirList']['log'])
        params['logWriter'] = writer

    return params['logWriter']

def SaveLog(params, FLAGS):

    writer = params['logWriter']
    writer.flush()


def RestoreGlobalCheckpoint(step, params, FLAGS, startup=False):

    checkpoint_name = FLAGS.experimentName+".s."+str(params['stage'])+".step."

    if startup:

        dir = params['dirList']['checkpoint'] + FLAGS.startCheckpointStage
        checkpointPath = dir + checkpoint_name + str(step) + '.pt'

    else:

        dir = params['dirList']['checkpoint'] + FLAGS.checkpointStage
        checkpointPath = dir + checkpoint_name + str(step) + '.pt'

    device = params['device']
    checkpoint = torch.load(checkpointPath, map_location=device)

    for key, module in params['modules'].items():
        module.load_state_dict(checkpoint[key])

    for key, optimizer in params['optimizers'].items():
        #debug
        # if key == 'p_optimizer_l0':
        #     continue
        # if key == 'p_optimizer_l1':
        #     continue

        optimizer.load_state_dict(checkpoint[key])

    params['epoch'] = checkpoint['epoch']
    params['stage'] = checkpoint['stage']

    if not params['dirList']['log']== checkpoint['logDir']:
        params['logWriter'] = None
        params['dirList']['log']= checkpoint['logDir']

    logging.debug('Load checkpoint from path :' + str(checkpointPath))
    return


def SaveGlobalCheckpoint(params, FLAGS):

    checkpoint_dir = params['dirList']['checkpoint'] + FLAGS.checkpointStage
    checkpoint_name = FLAGS.experimentName+".s."+str(params['stage'])+".step."

    last_checkpoints = LastCheckpoints(checkpoint_dir, FLAGS.experimentName)
    last_checkpoints.reverse()

    while len(last_checkpoints) > FLAGS.maxToKeep:
        to_delete = checkpoint_dir + last_checkpoints.pop()
        os.remove(to_delete)
        logging.debug('Remove the deprecated checkpoint :' + str(to_delete))


    state_dict = {}

    globalStep = params['globalStep']

    state_dict['globalStep'] = globalStep
    state_dict['epoch'] = params['epoch']
    state_dict['stage'] = params['stage']
    state_dict['logDir'] = params['dirList']['log']

    for key, module in params['modules'].items():
        state_dict[key] = module.state_dict()

    for key, optimizer in params['optimizers'].items():
        state_dict[key] = optimizer.state_dict()


    checkpointPath = checkpoint_dir + checkpoint_name + str(globalStep) + ".pt"

    torch.save(state_dict, checkpointPath)
    logging.debug('Save checkpoint to path :' + str(checkpointPath))


def NormalizeToOne(images):
    ori_shape = images.shape
    imgs = t_utility.StandardizeShape(images)
    batches, h, w, channels = imgs.shape
    for b in range(batches):
        for c in range(channels):
            img = imgs[b, :, :, c]

            min, max = img.min(), img.max()
            if max > min:
                img = (img - min) / (max - min)

            imgs[b, :, :, c] = img
    return np.reshape(imgs, ori_shape)


def Whiten(images, positive_constrain=False):
    ori_shape = images.shape
    imgs = t_utility.StandardizeShape(images)
    batches, h, w, channels = imgs.shape

    for b in range(batches):
        for c in range(channels):
            img = imgs[b, :, :, c]

            mean, std = np.mean(img), np.std(img)
            img = (img - mean) / std
            if (positive_constrain):
                min, max = img.min(), img.max()
                if max > min:
                    img = (img - min) / (max - min)
            imgs[b, :, :, c] = img

    return np.reshape(imgs, ori_shape)


def DataAugment(images, labels,
                resize=None,  # (width, height) tuple or None
                horizontal_flip=False,
                vertical_flip=False,
                rotate=0,  # Maximum rotation angle in degrees
                crop_probability=0,  # How often we do crops
                crop_min_percent=0.6,  # Minimum linear dimension of a crop
                crop_max_percent=1.,  # Maximum linear dimension of a crop
                mixup=0):  # Mixup coeffecient, see https://arxiv.org/abs/1710.09412.pdf
    if resize is not None:
        images = tf.image.resize_bilinear(images, resize)

    # My experiments showed that casting on GPU improves training performance
    if images.dtype != tf.float32:
        images = tf.image.convert_image_dtype(images, dtype=tf.float32)
        images = tf.subtract(images, 0.5)
        images = tf.multiply(images, 2.0)
    labels = tf.to_float(labels)

    with tf.name_scope('augmentation'):
        shp = tf.shape(images)
        batch_size, height, width = shp[0], shp[1], shp[2]
        width = tf.cast(width, tf.float32)
        height = tf.cast(height, tf.float32)

        # The list of affine transformations that our image will go under.
        # Every element is Nx8 tensor, where N is a batch size.
        transforms = []
        identity = tf.constant([1, 0, 0, 0, 1, 0, 0, 0], dtype=tf.float32)
        if horizontal_flip:
            coin = tf.less(tf.random_uniform([batch_size], 0, 1.0), 0.5)
            flip_transform = tf.convert_to_tensor(
                [-1., 0., width, 0., 1., 0., 0., 0.], dtype=tf.float32)
            transforms.append(
                tf.where(coin,
                         tf.tile(tf.expand_dims(flip_transform, 0), [batch_size, 1]),
                         tf.tile(tf.expand_dims(identity, 0), [batch_size, 1])))

        if vertical_flip:
            coin = tf.less(tf.random_uniform([batch_size], 0, 1.0), 0.5)
            flip_transform = tf.convert_to_tensor(
                [1, 0, 0, 0, -1, height, 0, 0], dtype=tf.float32)
            transforms.append(
                tf.where(coin,
                         tf.tile(tf.expand_dims(flip_transform, 0), [batch_size, 1]),
                         tf.tile(tf.expand_dims(identity, 0), [batch_size, 1])))

        if rotate > 0:
            angle_rad = rotate / 180 * math.pi
            angles = tf.random_uniform([batch_size], -angle_rad, angle_rad)
            transforms.append(
                tf.contrib.image.angles_to_projective_transforms(
                    angles, height, width))

        if crop_probability > 0:
            crop_pct = tf.random_uniform([batch_size], crop_min_percent,
                                         crop_max_percent)
            left = tf.random_uniform([batch_size], 0, width * (1 - crop_pct))
            top = tf.random_uniform([batch_size], 0, height * (1 - crop_pct))
            crop_transform = tf.stack([
                crop_pct,
                tf.zeros([batch_size]), top,
                tf.zeros([batch_size]), crop_pct, left,
                tf.zeros([batch_size]),
                tf.zeros([batch_size])
            ], 1)

            coin = tf.less(
                tf.random_uniform([batch_size], 0, 1.0), crop_probability)
            transforms.append(
                tf.where(coin, crop_transform,
                         tf.tile(tf.expand_dims(identity, 0), [batch_size, 1])))

        if transforms:
            images = tf.contrib.image.transform(
                images,
                tf.contrib.image.compose_transforms(*transforms),
                interpolation='BILINEAR')  # or 'NEAREST'

        def cshift(values):  # Circular shift in batch dimension
            return tf.concat([values[-1:, ...], values[:-1, ...]], 0)

        if mixup > 0:
            mixup = 1.0 * mixup  # Convert to float, as tf.distributions.Beta requires floats.
            beta = tf.distributions.Beta(mixup, mixup)
            lam = beta.sample(batch_size)
            ll = tf.expand_dims(tf.expand_dims(tf.expand_dims(lam, -1), -1), -1)
            images = ll * images + (1 - ll) * cshift(images)

            if (labels.shape[-1] > 1):
                lam = tf.expand_dims(lam, -1)

            labels = lam * labels + (1 - lam) * cshift(labels)

    return images, labels


class DataPreprocessor(torch.nn.Module):

    class StandardScaler(sklearn.preprocessing.StandardScaler):

        def to(self, device):
            self.mean_t = self.mean_t.to(device)
            self.scale_t = self.scale_t.to(device)

        def fit(self, X, y=None):
            sklearn.preprocessing.StandardScaler.fit(self, X, y)
            self.mean_t = torch.from_numpy(self.mean_).float()
            self.scale_t = torch.from_numpy(self.scale_).float()

        def transform(self, X, copy=None):
            if isinstance(X, np.ndarray):
                return sklearn.preprocessing.StandardScaler.transform(self, X, copy)
            else:  # torch

                sp = X.shape
                X = X.view(sp[0], -1)

                if self.with_mean:
                    X = X - self.mean_t

                if self.with_std:
                    X = X /self.scale_t
                return X

        def inverse_transform(self, X, copy=None):
            if isinstance(X, np.ndarray):
                return sklearn.preprocessing.StandardScaler.inverse_transform(X, copy)
            else:  # torch
                if copy:
                    X = X.copy()
                if self.with_std:
                    X = X * self.scale_t
                if self.with_mean:
                    X = X + self.mean_t
            return X

    class PCA(sklearn.decomposition.PCA):

        def to(self, device):
            self.mean_t = self.mean_t.to(device)
            self.components_t = self.components_t.to(device)
            self.explained_variance_t = self.explained_variance_t.to(device)

        def fit(self, X, y=None):
            sklearn.decomposition.PCA.fit(self, X, y)

            # debug
            # self.mean_.fill(0.0)
            self.mean_t = torch.from_numpy(self.mean_).float()
            self.components_t = torch.from_numpy(self.components_).float()
            self.explained_variance_t = torch.from_numpy(self.explained_variance_).float()

        def transform(self, X):
            if isinstance(X, np.ndarray):
                return sklearn.decomposition.PCA.transform(self, X)
            else:  # torch
                sp = X.shape
                X = X.view(sp[0], -1)
                if self.mean_ is not None:
                    X = X - self.mean_t
                X_transformed = X.matmul(self.components_t.t())
                if self.whiten:
                    X_transformed = X_transformed / torch.sqrt(self.explained_variance_t)

                return X_transformed

        def inverse_transform(self, X):
            if isinstance(X, np.ndarray):
                return sklearn.decomposition.PCA.inverse_transform(self, X)
            else:  # tf
                if self.whiten:
                    return (X.matmul((torch.sqrt(torch.unsqueeze(self.explained_variance_t, dim=-1))) *
                                 self.components_t)) + self.mean_t
                else:
                    return X.matmul(self.components_t) + self.mean_t

    def __init__(self, data, components=0.8, standard_scalar=True, pca=True, pca_whiten=False):

        super(DataPreprocessor, self).__init__()

        self.scaler = None
        self.pcaInstance = None

        x_std = data

        if standard_scalar:
            self.scaler = self.StandardScaler()
            self.scaler.fit(x_std)
            x_std = self.scaler.transform(x_std)
            self.scaler.maximum_output = max(x_std.max(), -x_std.min())

        # cov = np.cov(x_std.T)
        # eigVals, eigVecs = np.linalg.eig(cov)
        # self.eigVecs = eigVecs[np.argsort(eigVals)]

        if pca:
            self.pcaInstance = self.PCA(n_components=components, whiten=pca_whiten)
            self.pcaInstance.fit(x_std)

        return

    def transform(self, data):

        preOut = data
        if self.scaler is not None:
            preOut = self.scaler.transform(preOut)
            # data /= self.maximum

        if self.pcaInstance is not None:
            preOut = self.pcaInstance.transform(preOut)

        return preOut

    def forward(self, data):
        return self.transform(data)

    def to(self, device):

        super().to(device)

        if self.scaler is not None:
            self.scaler.to(device)
        if self.pcaInstance is not None:
            self.pcaInstance.to(device)

    def inverse_transform(self, data):

        preOut = data

        if not self.pcaInstance == None:
            preOut = self.pcaInstance.inverse_transform(preOut)

        if not self.scaler == None:
            preOut = self.scaler.inverse_transform(preOut)

        return preOut


class ImgArrayDataSet(torch.utils.data.Dataset):

    def __init__(self, data, targets, transform = None, target_transform = None):

        super().__init__()
        self.data = t_io.ToLDR(data)
        self.targets = targets
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

        img, target = self.data[index], self.targets[index]

        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

class DualDataSet(torch.utils.data.Dataset):

    def __init__(self, a_inputs, a_targets, b_inputs, b_targets, a_transforms=None, b_transforms=None):

        super().__init__()

        self.a_inputs = a_inputs
        self.a_targets = a_targets
        self.a_transforms = a_transforms

        self.b_inputs = b_inputs
        self.b_targets = b_targets
        self.b_transforms = b_transforms

        def Tile(inputs, times):
            sp = [1] * len(inputs.shape)
            sp[0] = times
            return np.tile(inputs, sp)

        self.a_inputs = Tile(self.a_inputs, 2)
        self.a_targets = Tile(self.a_targets, 2)
        self.a_inputs = self.a_inputs[5000:,...]
        self.a_targets = self.a_targets[5000:,...]


        if a_inputs.shape[0] < b_inputs.shape[0]:
            times = int(b_inputs.shape[0] / a_inputs.shape[0])
            self.a_inputs = Tile(a_inputs, times)
            self.a_targets = Tile(a_targets, times)


        if b_inputs.shape[0] < a_inputs.shape[0]:
            times = int(a_inputs.shape[0] / b_inputs.shape[0])
            self.b_inputs = Tile(b_inputs, times)
            self.b_targets = Tile(b_targets, times)

        self.a_targets = torch.from_numpy(self.a_targets)
        self.b_targets = torch.from_numpy(self.b_targets)


    def __len__(self):
        return 2 * self.a_inputs.shape[0]

    def __getitem__(self, idx):

        if idx < self.a_inputs.shape[0]:
            a_x = self.a_inputs[idx, ...]
            a_y = self.a_targets[idx, ...]


            if self.a_transforms is not None:
                img = Image.fromarray((a_x * 255).astype('uint8'))
                a_x = self.a_transforms(img)

            return a_x, a_y

        else:
            idx = idx - self.a_inputs.shape[0]

            b_x = self.b_inputs[idx, ...]
            b_y = self.b_targets[idx, ...]

            if self.b_transforms is not None:
                img = Image.fromarray((b_x * 255).astype('uint8'))
                b_x = self.b_transforms(img)

            return b_x, b_y



