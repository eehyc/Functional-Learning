from absl import flags

FLAGS = flags.FLAGS

# --loadPropagation --learningRate 1e-4 --lossFunc MAPE --trainingData TuningDevice --layerAction ['evaluate','pass'] --trainingFilter {'pass':['*ScopicWeight*','*Factor*']}

# Name
flags.DEFINE_string('experimentName', 'MNIST', 'Name of current experiment')
flags.DEFINE_string('oDevice', '2LT_OS-2', 'The name of the oDevice.')

flags.DEFINE_integer('dWorkers', 4, 'Number of dataloader workers.')

# Configuration
flags.DEFINE_boolean('activateHardware', False, 'Inidcate whether to activate camera and displays. ')
flags.DEFINE_boolean('configureCamera', False, 'Inidcate whether to configure camera.')

# Socket
flags.DEFINE_string('hardwareSocket', '',
                    "Socket type: 'client' or 'server'. NOTE: server is the machine controlling the target optical device.")
flags.DEFINE_string('hardwareAddress', '', 'The server address for hardware socket.')
flags.DEFINE_integer('hardwarePort', 19745, 'Port for hardware socket.')

flags.DEFINE_integer('dynamicRange', 12, 'The dynamic range of captured image. ')
flags.DEFINE_boolean('calibrateLineality', False, 'Indicate whether to configure lineality. ')
flags.DEFINE_boolean('generatePropagation', False, 'Indicate whether to generate propagation from scratch.')

# The size of input layer'
flags.DEFINE_string('inputShape', '[3,32,32]', 'Shape of input data of networks, (h,w,c) by default.')
flags.DEFINE_string('channel', '[1,1,1]', 'The active RGB channel. ')
flags.DEFINE_integer('oneHotCount', 10, 'The number of output digits.')

# Directory
flags.DEFINE_string('projectDir', '', 'Path to the project.')
flags.DEFINE_string('inputDir', '', 'Path to the input, default value is projectDir/input.')
flags.DEFINE_string('outputDir', '', 'Path to the output, default value is projectDir/output.')

# Network setting
flags.DEFINE_string('lossFunc', 'Zero',
                    'Loss function to train on. Choose from L1, L2, MAPE, RELMSE, LMLS, SSIM or CrossEntropy.')
flags.DEFINE_string('errorList', '[]',
                    'Error function to evaluate. A list with elements choosen from PredictAccuracy, L1, L2, MAPE, RELMSE, LMLS, SSIM or CrossEntropy.')
flags.DEFINE_integer('seed', -1, 'Seed for weight initialization.')
flags.DEFINE_integer('batchSize', 128, 'Batch size for operation.')
flags.DEFINE_boolean('closeShuffleData', False, 'Indicate wheter to shuffle data.')
flags.DEFINE_float('regularParameter', 0, 'Regularization to prevent over-fitting')
# flags.DEFINE_float('regularParameter', 0.00001, 'Regularization to prevent over-fitting')
flags.DEFINE_boolean('batchNormalization', False, 'Indicate wheter to apply batch normalization.')

# Control
flags.DEFINE_string('task', "['train', 'gatherClient']", 'Indicate the task to performs, e.g., training / test.')

# Restore
flags.DEFINE_boolean('restore', False, 'Load models.')
flags.DEFINE_boolean('continueLastStage', False, 'Continue the training of last stage.')
flags.DEFINE_string('checkpointStage', '', 'The path to save the checkpoints of current stage.')
flags.DEFINE_string('startCheckpointStage', '', 'If the current stage is empty, search the startup stage.')
flags.DEFINE_integer('startStep', 145700,
                     'Step to start from. Valid only if restore is set to True. If -1 automatically serch latest step.')
flags.DEFINE_integer('startStage', 0,
                     'Stage to start from. Valid only if restore is set to True and startStep is set to -2.')
flags.DEFINE_integer('maxToKeep', 500, 'Maximum number of checkpoints to keep.')

flags.DEFINE_string('summaryStage', '', 'The path to save the sumarries of current stage.')

flags.DEFINE_string('trainingFilter', '',
                    'The filter to indicate training variables. An example : {"pass":["*layer0*", "*layer1*"],"fail":["*Scopic*"]} ')
flags.DEFINE_string('layerAction', '["evaluate","evaluate"]',
                    'The filter to indicate layer actors. An example : ["evaluate","pass"] ')

flags.DEFINE_string('data', 'MNIST', 'The data used for training.')
flags.DEFINE_string('GT', '', 'Specify GT to special task.')
flags.DEFINE_boolean('PCA', False, 'Indicate wheter to apply PCA.')
flags.DEFINE_boolean('dataAugmentation', False, 'Indicate wheter to apply data augmentation.')

# Hyper parameter
flags.DEFINE_float('learningRate', 1e-3, 'Learning rate.')
flags.DEFINE_float('decayRate', 0, 'Decay speed of learning rate. 0 means using specified learning rate.')
flags.DEFINE_integer('checkpointStep', 10000, 'Number of step before saving checkpoint and evaluation.')
flags.DEFINE_boolean('skipEvaluation', False, 'Skip initializing and conducting evaluation.')
flags.DEFINE_boolean('skipTrain', False, 'Skip initializing and conducting training.')
flags.DEFINE_integer('summaryStep', 5000, 'Number of step before saving summary.')
flags.DEFINE_integer('maximumRecords', 10, 'The maximum records to keep.')
flags.DEFINE_integer('maxStep', 50000000,
                     'A ridiculous number of training steps. User can (and should) quit before this is reached.')
flags.DEFINE_integer('maxStage', 20, 'Maximum stages.')
flags.DEFINE_boolean('writeMetaGraph', False, 'Indicate wheter to export meta graph to checkpoint. ')


flags.DEFINE_integer('numOLayers', 3, 'Number of optical layers.')


import logging
import sklearn
import numpy as np
import t_io
import ast
import platform
import threading


import oDevice_i_2L_i
import oLayer_FC_i_xLT_i
import trainer_PZ

import layer_PRUNE_i_DIG_oh
import layer_BN_c_DIG_c
import layer_PLANE_c_DIG_c

import t_socket
import t_data
import t_utility
import t_layer
import sys
import t_tis_camera as t_camera
import t_calibration

import torch

import torchvision
import torchvision.transforms as transforms

import t_opengl_display
import some_common_functions


def RunHardwareServerThenExit(params, FLAGS):


    server = t_socket.ThreadingSocket('server', FLAGS.hardwareAddress, FLAGS.hardwarePort)

    server.run()

    models = params['models']

    o_model = models['o_model']
    oLayer = o_model[0]

    while True:
        data = server.recv_queue.get()

        LCs = data['LCs']
        input_attention = data['input_attention']
        output_attention = data['output_attention']
        inputs = data['inputs']
        outputs = oLayer.sense(inputs, LCs, input_attention, output_attention)

        send_data = {}
        send_data['outputs'] = outputs

        server.send_queue.put(send_data)

    exit(0)

# Initialize neural network.
def InitializeModules(params, FLAGS):

    modules = {}
    device = params['device']

    c, h, w = FLAGS.inputShape[0], FLAGS.inputShape[1], FLAGS.inputShape[2]


    full_layers = []
    o_layers = []
    a_layers = []
    layer_idx = 0



    for l in range(FLAGS.numOLayers):

        # activation layer
        name = 'pl_' + str(l)
        p_layer = layer_BN_c_DIG_c.layer_BN_c_DIG_c(3, name).to(device)

        modules[name] = p_layer
        a_layers.append(p_layer)
        full_layers.append(p_layer)

        if l == 0:
            p_layer.skip = True

        # optical layer
        name = 'fc_' + str(l)
        layer = oLayer_FC_i_xLT_i.oLayer_FC_i_xLT_i(params['oDevices'][FLAGS.oDevice], params, FLAGS, name)

        modules[name] = layer
        o_layers.append(layer)
        full_layers.append(layer)


    # one hot layer
    oh_name = 'oh'
    oh_layer = layer_PRUNE_i_DIG_oh.layer_PRUNE_i_DIG_oh((np.sum(layer.channel), h, w), FLAGS.oneHotCount,
                                                         [4, 4],
                                                         params, FLAGS, oh_name).to(device)
    modules[oh_name] = oh_layer

    full_layers.append(oh_layer)


    layer_idx += 1


    for module in modules.values():
        module.to(params['device'])

    models = {}
    optimizers = {}

    full_model = t_layer.Sequential(*(full_layers))
    models['full_model'] = full_model
    models['p_model'] = full_model



    full_model.requires_grad_(True)
    full_model.requires_grad_z(False)
    optimizers['p_optimizer'] = torch.optim.Adam(filter(lambda p:p.requires_grad, full_model.parameters()), lr=FLAGS.learningRate)
    full_model.requires_grad_(True)


    for o in range(len(o_layers)):

        z_model = t_layer.Sequential(a_layers[o], o_layers[o])

        z_model.requires_grad_(True)
        z_model.requires_grad_p(False)
        optimizers['z_optimizer_l'+str(o)] = torch.optim.Adam(filter(lambda p:p.requires_grad, z_model.parameters()), lr=FLAGS.learningRate)
        z_model.requires_grad_(True)


        models['z_model_l'+str(o)] = z_model

        if o == len(o_layers) - 1:
            z_model = t_layer.Sequential(a_layers[o],o_layers[o],oh_layer)

        z_model.requires_grad_(False)
        z_model[1].requires_grad_p(True)
        optimizers['p_optimizer_l'+str(o)] = torch.optim.Adam(filter(lambda p:p.requires_grad, z_model.parameters()), lr=FLAGS.learningRate)
        z_model.requires_grad_(True)
        # optimizers['p_optimizer_l' + str(o)] = torch.


    # modules['last2'] = layer_MLP_c_DIG_c.layer_MLP_c_DIG_c(FLAGS.inputShape, FLAGS.inputShape).to(device)
    modules['last'] = layer_PLANE_c_DIG_c.layer_PLANE_c_DIG_c(3).to(device)
    # modules['last'].regular = False

    params['modules'] = modules
    params['models'] = models
    params['optimizers'] = optimizers

    return


# Initialize hardware.
def InitializeDevices(params, FLAGS):

    if FLAGS.activateHardware == True:

        # Hardware.
        components = {}

        # Activate camera.
        if t_camera.Active() is True:
            camera = t_camera.Instance(FLAGS, params, FLAGS.configureCamera, False)
        else:
            logging.error('Tried to start TIS camera in an unprepared device.')
            sys.exit(0)

        # Read output plane by the camera.
        output_plane = camera.interface()

        # Use openGL to control input plane, lc0, lc1.
        display_list = [None, 'input', 'lc0', 'lc1']
        displays = t_opengl_display.Instance(display_list)

        input_plane = displays.interface('input')
        lc0 = displays.interface('lc0')
        lc1 = displays.interface('lc1')
        displays.refreshAll()
        displays.focus()

        # Drop useless pixels from the input plane, output plane, lc0, lc1.
        input_plane, lc0, lc1, output_plane = t_calibration.CalibrateCrop(input_plane, lc0, lc1, output_plane)

        # Measure the background noise of the output plane, not really necessary
        # bg = np.zeros(output_plane.hwc())
        # for i in range(100):
        #     bg += t_io.ToHDR(output_plane.sense())
        # bg /= 100
        #
        # output_plane.bg = bg

        # Replace the gamma LUT with a measured linear LUT for LC panels.
        if FLAGS.calibrateLineality:

            # t_calibration.MeasureLCResponse(params['bg'], input_plane, lc0, lc1, output_plane, displays, [32,32,32], channels)

            input_plane.emit(np.zeros(input_plane.hwc()))
            lc0.emit(np.zeros(lc0.hwc()))
            lc1.emit(np.zeros(lc1.hwc()))
            input_plane.delay()
            bg = output_plane.sense()

            func1 = np.ones
            func2 = np.ones

            lc0.emit(func1(lc0.hwc()))
            lc1.emit(func2(lc1.hwc()))

            t_calibration.CalibrateSpectrumCorrelation(bg, output_plane, input_plane, displays)

            for c in range(3):
                if FLAGS.channel[c] == 0:
                    continue

                ch = [0, 0, 0]
                ch[c] = 1

                lc0.emit(func2(lc0.hwc()) * ch)
                lc1.emit(func2(lc1.hwc()) * ch)
                t_calibration.CalibrateDisplayLinearlity(bg, output_plane, input_plane, displays, 'input', ch)

                input_plane.emit(func1(input_plane.hwc()) * ch)
                lc1.emit(func2(lc1.hwc()) * ch)
                t_calibration.CalibrateDisplayLinearlity(bg, output_plane, lc0, displays, 'lc0', ch)

                input_plane.emit(func1(input_plane.hwc()) * ch)
                lc0.emit(func2(lc0.hwc()) * ch)
                t_calibration.CalibrateDisplayLinearlity(bg, output_plane, lc1, displays, 'lc1', ch)

            print("Write LUT.")
            displays.writeProperty(params, FLAGS)
        else:
            displays.readProperty(params, FLAGS)


        components['input'] = input_plane
        components['lc0'] = lc0
        components['lc1'] = lc1
        components['output'] = output_plane
        params['components'] = components

    # Assemble the raw hardware into light field neural network devices
    devices = {}
    spec = oDevice_i_2L_i.MeasurementSpec(params, FLAGS, FLAGS.oDevice, 'input', ['lc0', 'lc1'], 'output')

    spec['iVirtualHeightP'] = 32
    spec['iVirtualWidthP'] = 32

    spec['lVirtualHeightP'] = 32
    spec['lVirtualWidthP'] = 32

    spec['oVirtualHeightP'] = 32
    spec['oVirtualWidthP'] = 32

    # spec = device_i_2L_i.SimulationSpec(params, FLAGS, FLAGS.device)

    d = oDevice_i_2L_i.ODevice_i_2L_i(spec, params, FLAGS)
    d.activateHardware(params, FLAGS)
    d.initializePropagation(params, FLAGS)
    d.initializeTensor(params, FLAGS)

    devices[d.name] = d


    params['oDevices'] = devices

    return 0


def InitializeDataset(params, FLAGS):

    shuffle_data = not FLAGS.closeShuffleData

    if FLAGS.data == 'MNIST':

        transform = transforms.Compose([
            transforms.Pad((2, 2, 2, 2)),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.view(1, 32, 32).expand(3, -1, -1))
        ])

        train_set = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        test_set = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

        train_loader = torch.utils.data.DataLoader(train_set, batch_size=FLAGS.batchSize, shuffle=shuffle_data,
                                                   num_workers=FLAGS.dWorkers, pin_memory=True, drop_last=True)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=FLAGS.batchSize, shuffle=False,
                                                  num_workers=FLAGS.dWorkers, pin_memory=True)

        # debug
        # test_loader = train_loader

        def convert_to_cuda(train_set):

            temp = train_set.data.numpy().astype(np.float32)
            temp /= 255.

            temp = temp.reshape((-1, 1, 28, 28))

            temp = np.pad(temp, [[0, 0], [0, 0], [2, 2], [2, 2]], mode='constant')
            temp = t_utility.UpScale(temp, [1, 3, 1, 1])

            train_set.data = torch.from_numpy(temp).cuda()

            train_set = torch.utils.data.TensorDataset(train_set.data, train_set.targets.cuda())
            return train_set

        if 'Windows' in platform.system():
            train_loader = torch.utils.data.DataLoader(convert_to_cuda(train_set), batch_size=FLAGS.batchSize,
                                                       shuffle=shuffle_data, num_workers=0, drop_last=True)

            test_loader = torch.utils.data.DataLoader(convert_to_cuda(test_set), batch_size=FLAGS.batchSize,
                                                      shuffle=False, num_workers=0)



    elif FLAGS.data == 'CIFAR10':

        shuffle_data = not FLAGS.closeShuffleData

        transform = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize(mean=(0, 0, 0), std=(1, 1, 1))
        ])

        train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)


        #############################################################################
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=FLAGS.batchSize, shuffle=shuffle_data,
                                                   num_workers=FLAGS.dWorkers, pin_memory=True, drop_last=True)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=FLAGS.batchSize, shuffle=False,
                                                  num_workers=FLAGS.dWorkers, pin_memory=False)


    # pca
    if FLAGS.PCA == True:

        data = []
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            data.append(inputs)


        data = torch.cat(data, dim=0)
        data = data.view(data.shape[0], -1)

        pca = t_data.DataPreprocessor(data, components=12, standard_scalar=False, pca_whiten=False)

        pca.to(params['device'])
        params['PCA'] = pca
    else:
        params['PCA'] = None


    params['p_dataset'] = (train_loader, test_loader)

    return params


def RunLocal(params, FLAGS):

    oDevice = params['oDevices'][FLAGS.oDevice]
    oDevice.initializeTensor(params, FLAGS, False)

    if FLAGS.hardwareSocket == 'server':
        RunHardwareServerThenExit(params, FLAGS)
        return

    step = params['globalStep']

    logging.info('Step :' + str(step))
    # Recover checkpoint
    if FLAGS.restore:
        logging.info('Restoring Model... ')
        t_data.RestoreGlobalCheckpoint(step, params, FLAGS, True)
        logging.info('Done.')


    # Train and Validation Loop
    if 'train' in FLAGS.task:

        if step < FLAGS.maxStep:
            # Train
            if step == 1:
                logging.info("\nStarting training...")
            else:
                logging.info("\nResuming training...")

            LocalTrain(params, FLAGS)

    return 0


def DesignData(params, FLAGS):
    input_set = []
    target_set = []
    for i in range(32-4):
        img = np.zeros((3, 32, 32),dtype=np.float32)
        j = i
        j = 16
        img[:,:,j:j+4] = 1
        input_set.append(img)
        target_set.append(img)

        img = np.zeros((3,32,32),dtype=np.float32)
        img[:,j:j+4,:]=1
        input_set.append(img)
        target_set.append(img)

    input_set = np.stack(input_set)
    target_set = np.stack(target_set)


    dataset = torch.utils.data.TensorDataset(torch.from_numpy(input_set), torch.from_numpy(target_set))
    o_loader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=False, num_workers=4)
    params['p_dataset'] = (o_loader, o_loader)

def ConfigureStage(stage, pre_stage_context, params, FLAGS):

    optimizers = params['optimizers']
    models = params['models']
    modules = params['modules']
    device = params['device']

    p_train_loader, p_test_loader = params['p_dataset']

    class Config():

        def __init__(self):

            self.ZDataUpdation = t_utility.SparsityFunction(0, 100 * FLAGS.numOLayers, 1, 0.01)
            self.SaveZData = False

    config = Config()



    def FineTuneStage():
        config.SaveZData = True

        p_model = models['p_model']
        p_optimizer = optimizers['p_optimizer']
        epoch_callbacks = []

        for i in range(FLAGS.numOLayers):
            p_model[2 * i + 1].noise_level = 0

        trainers = []
        for l in range(0, FLAGS.numOLayers):
            z_model = models['z_model_l' + str(l)]
            z_optimizer = optimizers['z_optimizer_l' + str(l)]

            trainer = trainer_PZ.Trainer_PZ(params,
                                            z_model, torch.nn.L1Loss(), z_optimizer,
                                            p_model, torch.nn.CrossEntropyLoss(), p_optimizer,
                                            p_train_loader, p_test_loader, t_utility.Accuracy())

            trainers.append(trainer)

        return [FLAGS.numOLayers * 100, epoch_callbacks, trainers, config]


    if stage <= 1:


        p_model = models['p_model']
        p_optimizer = optimizers['p_optimizer']
        epoch_callbacks = []

        trainers = []
        for l in range(FLAGS.numOLayers):
            z_model = models['z_model_l' + str(l)]
            z_optimizer = optimizers['z_optimizer_l' + str(l)]
            #p_optimizer = optimizers['p_optimizer_l' + str(l)]

            trainer = trainer_PZ.Trainer_PZ(params,
                                            z_model, torch.nn.L1Loss(), z_optimizer,
                                            p_model, torch.nn.CrossEntropyLoss(), p_optimizer,
                                            p_train_loader, p_test_loader, t_utility.Accuracy())


            trainers.append(trainer)


        return [FLAGS.numOLayers * 100, epoch_callbacks, trainers, config]

    elif stage == 2:

        p_model = models['p_model']
        p_optimizer = optimizers['p_optimizer']
        epoch_callbacks = []


        prune_layer = p_model[-1]
        prune_layer.setupSparsityFunction(t_utility.SparsityFunction(0, FLAGS.numOLayers * 50, 0., 1.0 - 4.0 / prune_layer.dotSize, 1, 5))
        epoch_callbacks.append(prune_layer.updatePrune)


        trainers = []
        for l in range(0, FLAGS.numOLayers):

            z_model = models['z_model_l' + str(l)]
            z_optimizer = optimizers['z_optimizer_l' + str(l)]
            #p_optimizer = optimizers['p_optimizer_l' + str(l)]

            trainer = trainer_PZ.Trainer_PZ(params,
                                            z_model, torch.nn.L1Loss(), z_optimizer,
                                            p_model, torch.nn.CrossEntropyLoss(), p_optimizer,
                                            p_train_loader, p_test_loader, t_utility.Accuracy())


            trainers.append(trainer)

        return [FLAGS.numOLayers * 100, epoch_callbacks, trainers, config]

    elif stage == 3:

        return FineTuneStage()

    elif stage >= 3 and stage < 20:
        if pre_stage_context is None:
            pre_stage_context = FineTuneStage()
        pre_stage_context[-1].ZDataUpdation = t_utility.SparsityFunction(0, 1, 0.01, 0.01)
        return pre_stage_context
    else:

        if pre_stage_context is None:
            pre_stage_context = FineTuneStage()
        pre_stage_context[-1].ZDataUpdation = t_utility.SparsityFunction(0, 1, 0.01, 0.01)

        # freeze previous layers
        trainers = pre_stage_context[2]

        logging.info('Freezing previous layers...')

        if stage == 9:
            for i in range(100):
                logging.info(i)
                trainers[0].run(log=False,Z=False)
                trainers[0].evaluate(log=False)


        for i in range(FLAGS.numOLayers - 1):
            trainers[i].freeze = True
            trainers[i].z_model.requires_grad_(False)

        # change last trainer
        last_trainer = trainers[-1]
        p_model = models['p_model']


        pre_model = t_layer.Sequential(*(p_model[:2*(FLAGS.numOLayers - 1)]))
        pre_model.requires_grad_(False)

        last_model = t_layer.Sequential(*(p_model[2*(FLAGS.numOLayers - 1):]))
        last_trainer.p_model = last_model


        last_trainer.p_train_loader = t_utility.ConvertLoader(p_train_loader, pre_model, True, FLAGS.batchSize, FLAGS.dWorkers)
        last_trainer.p_test_loader = t_utility.ConvertLoader(p_test_loader, pre_model, False, FLAGS.batchSize, FLAGS.dWorkers)


        return pre_stage_context





class ZDataThread(threading.Thread):

    def __init__(self, inputs, labels, z_model):
        super().__init__()

        # self.success = False

        self.inputs = inputs
        self.labels = labels
        self.timeout = 10.0
        self.z_model = z_model


        if self.z_model is not None:
            with torch.no_grad():
                self.record_p_parameters = {}

                for key, par in t_utility.GatherDict(self.z_model, 'p_parameters').items():
                    self.record_p_parameters[key] = par.detach().clone()


    def start(self):
        self.success = False
        super().start()

    def stop(self):
        self._stop_event.set()

    def composeLoaders(self):

        def ComposeLoader(inputs, targets):
            set = torch.utils.data.TensorDataset(inputs, targets)
            loader = torch.utils.data.DataLoader(set, batch_size=FLAGS.batchSize, shuffle=False,
                                                 num_workers=FLAGS.dWorkers)
            return loader

        output_loader, label_loader = None, None

        if not self.outputs is None:
            inputs = t_io.ImgConvertTorchAndNumpy(self.inputs)
            outputs = t_io.ImgConvertTorchAndNumpy(self.outputs)
            output_loader = ComposeLoader(inputs, outputs)
            task_loader = ComposeLoader(outputs, self.labels)

        return output_loader, task_loader

    def run(self):

        try:

            self.outputs = None

            if not self.inputs is None:
                self.outputs = self.z_model.sense(self.inputs)


        except BaseException as e:
            # logging.error(e)
            self.success = False
            return

        self.success = True

    def wait(self, time_out):

        if not self.is_alive() and self.success == False:
            return False

        self.join(timeout=time_out)

        if not self.is_alive() and self.success == True:
            return True
        else:
            try:
                t_utility.stop_thread(self)
                # c_data_thread.join()
            except Exception as ex:
                pass
            logging.error('Stopped c-thread after timeout.')
            return False

def OneStage(stage_id, stage_context, params, FLAGS):

    target_epochs, epoch_callbacks, all_trainers, stage_config = stage_context

    trainers = []
    for trainer in all_trainers:
        if trainer.freeze == False:
            trainers.append(trainer)

    log_writer = t_data.GetLogWriter(params, FLAGS)
    log_space = 's' + str(stage_id) + '/'

    for trainer in trainers:
        trainer.log_writer = log_writer
        trainer.log_space = log_space


    train_time_out = 100
    num_train_batch = 1
    num_test_batch = 1

    gather_id = 0
    train_id = 0

    epoch = params['epoch']

    sp = stage_config.ZDataUpdation

    for e in range(1, target_epochs + 1):

        if e == 1:
            train_inputs, train_labels = some_common_functions.FetchDeviceInputs(trainers[gather_id].p_train_loader, num_train_batch)
            z_data_thread = ZDataThread(train_inputs, train_labels, trainers[gather_id].z_model)
            z_data_thread.start()

        logging.debug('Begin epoch %d' % (e))

        while 1:
            if z_data_thread.wait(train_time_out):
                trainers[gather_id].z_dataset.random_update(t_io.ImgConvertTorchAndNumpy(z_data_thread.inputs),
                                                            t_io.ImgConvertTorchAndNumpy(z_data_thread.outputs),
                                                            z_data_thread.record_p_parameters, update_ratio=sp(e))

                train_id = gather_id
                gather_id = (gather_id + 1) % len(trainers)

                if gather_id == 0:
                    train_inputs, test_labels = some_common_functions.FetchDeviceInputs(trainers[gather_id].p_train_loader, num_train_batch)
                else:
                    train_inputs = z_data_thread.outputs
                    train_labels = z_data_thread.labels

                z_data_thread = ZDataThread(train_inputs, train_labels, trainers[gather_id].z_model)
                break
            else:
                z_data_thread = ZDataThread(z_data_thread.inputs, z_data_thread.labels, z_data_thread.z_model)
                logging.critical('restart the failed c data thread')
                z_data_thread.start()

        if e < target_epochs + 1:
            z_data_thread.start()

        # logging.info('C-data thread started.')
        logging.info('Start training epoch %d.' % (e))

        # training
        trainers[train_id].step = params['globalStep']

        trainers[train_id].run(e)

        with torch.no_grad():
            for call_back in epoch_callbacks:
                call_back()

        trainers[train_id].evaluate(e)

        trainers[train_id].saveState(epoch, train_id)

        params['globalStep'] = trainers[train_id].step
        epoch += 1

        # pz evaluation
        if 0 and (e) % (10 * FLAGS.numOLayers) == 0:

            num_test_batch = 1
            test_timeout = 60

            test_timeout *= num_test_batch

            inputs, labels = some_common_functions.FetchDeviceInputs(trainers[0].p_test_loader, num_test_batch)


            fail_flag = False
            for l in range(len(trainers)):


                pz_data_thread = ZDataThread(inputs, labels, trainers[l].z_model)
                pz_data_thread.start()

                if pz_data_thread.wait(test_timeout) == False:
                    fail_flag = True
                    break

                inputs = pz_data_thread.outputs
                labels = pz_data_thread.labels

            if fail_flag == False:
                output_loader, label_loader = pz_data_thread.composeLoaders()

                error = t_utility.EvaluateErrors(label_loader, trainers[-1].p_model[-1], trainers[-1].p_error_func).cpu().numpy()
                name = str(trainers[-1].p_error_func)
                logging.critical('PZ evaluation | ' + name + ' ' + str(error))

                log_writer.add_scalar(log_space + 'test/PZ/' + name, error, e)


def LocalTrain(params, FLAGS):


    epoch = params['epoch']
    stage = params['stage']

    stage_context = None
    first_stage = True

    while stage <= FLAGS.maxStage:  ################## C ##################

        logging.info('Begin stage %d' % (stage))

        stage_context = ConfigureStage(stage, stage_context, params, FLAGS)
        target_epochs, epoch_callbacks, trainers, stage_config = stage_context

        if first_stage == True:
            for i in range(len(trainers)):
                trainers[i].LoadZData(stage, i, params, FLAGS)
            first_stage = False


        OneStage(stage, stage_context, params, FLAGS)

        stage += 1
        epoch += target_epochs

        params['epoch'] = epoch
        params['stage'] = stage


        t_data.SaveGlobalCheckpoint(params, FLAGS)
        t_data.SaveLog(params, FLAGS)

        if stage_config.SaveZData == True:
            for i in range(len(trainers)):
                trainers[i].SaveZData(stage, i, params, FLAGS)




def ComposeDataLoader(train_inputs, train_targets, test_inputs, test_outputs, test_labels, params, FLAGS):
    if not train_inputs is None:
        train_set = torch.utils.data.TensorDataset((train_inputs), (train_targets))
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=FLAGS.batchSize, shuffle=True,
                                                   num_workers=FLAGS.dWorkers)
    else:
        train_loader = None

    if not test_inputs is None:
        test_set = torch.utils.data.TensorDataset((test_inputs), (test_outputs))
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=FLAGS.batchSize, shuffle=True,
                                                  num_workers=FLAGS.dWorkers)

        cow_set = torch.utils.data.TensorDataset((test_inputs), (test_labels))
        cow_loader = torch.utils.data.DataLoader(cow_set, batch_size=FLAGS.batchSize, shuffle=True,
                                                 num_workers=FLAGS.dWorkers)
    else:
        test_loader = None
        cow_loader = None

    return train_loader, test_loader, cow_loader


def ReadDeviceOutput(params, FLAGS):
    location = params['dirList']['checkpoint'] + FLAGS.checkpointStage
    location = location + 's.' + str(params['stage']) + '.device.npz'

    logging.info('Load device data from : ' + str(location))

    data = np.load(location)

    train_inputs = data['train_inputs']
    train_targets = data['train_targets']
    test_inputs = data['test_inputs']
    test_outpus = data['test_outputs']

    data.close()

    return train_inputs, train_targets, test_inputs, test_outpus


def main(argv):

    logging.basicConfig(level=logging.DEBUG, format="%(asctime)s %(levelname)s [%(filename)s:%(lineno)d]: %(message)s")

    params = {}

    params['globalStep'] = 0
    params['epoch'] = 0
    params['stage'] = 0

    if torch.cuda.is_available():
        params['device'] = torch.device("cuda")
        torch.backends.cudnn.benchmark = True
        logging.debug('PyTorch is using GPU ' + str(torch.cuda.current_device()))
    else:
        params['device'] = torch.device("cpu")

    logging.critical(params['device'])

    if FLAGS.seed >= 0:
        np.random.seed(FLAGS.seed)
        torch.manual_seed(FLAGS.seed + 9)

    FLAGS.inputShape = ast.literal_eval(FLAGS.inputShape)

    FLAGS.channel = ast.literal_eval(FLAGS.channel)
    FLAGS.layerAction = ast.literal_eval(FLAGS.layerAction)
    FLAGS.errorList = ast.literal_eval(FLAGS.errorList)
    FLAGS.task = ast.literal_eval(FLAGS.task)

    t_data.InitializeResultsFolder(params, FLAGS)
    t_data.InitializeStep(params, FLAGS)


    logging.info('Initializing device ...')
    InitializeDevices(params, FLAGS)
    InitializeModules(params, FLAGS)
    InitializeDataset(params, FLAGS)
    logging.info('Done.')


    RunLocal(params, FLAGS)


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG, format="%(asctime)s %(levelname)s [%(filename)s:%(lineno)d]: %(message)s")

    FLAGS(sys.argv)
    main(sys.argv)


