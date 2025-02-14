import torch
import logging
import t_utility
import random
import os
import numpy as np
from os import path



class Trainer_PZ(object):

    class BatchedDataset(object):

        def __init__(self, num_batches, device):

            self.device = device
            self.n_batches = num_batches
            self.shuffled_hash = list(range(self.n_batches))
            random.shuffle(self.shuffled_hash)
            self.batched_buffer = [None] * self.n_batches

            self.my_batch_idx = 0

            self.init_state = True

            self.log_writer = None


        def epoch(self):
            random.shuffle(self.shuffled_hash)

        def batch(self, batch_idx):
            return self.batched_buffer[batch_idx]

        def update(self, batch_idx, inputs, targets):

            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            self.batched_buffer[batch_idx] = (inputs, targets)


        def random_update(self, inputs, targets, p_parameters, update_batch=None, update_ratio=None):
           # update_batch=-1
            if update_batch is not None:
                n_times = update_batch
            elif update_ratio is not None:
                n_times = min(self.n_batches, round(update_ratio * self.n_batches))
                if update_ratio > 0:
                    n_times = max(n_times, 1)
            else:
                n_times = -1


            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            insert = ((inputs, p_parameters), targets)

            if self.init_state == True or n_times == -1:
                for i in range(self.n_batches):
                    self.batched_buffer[i] = insert
                self.init_state = False

            else:
                for i in range(n_times):
                    batch_idx = random.randint(0, self.n_batches - 1)

                    self.batched_buffer[batch_idx] = insert



    def SaveZData(self, stage, id, params, FLAGS):

        checkpoint_dir = params['dirList']['checkpoint'] + FLAGS.checkpointStage
        file_name = checkpoint_dir + 'ZData.' + str(id) + '.s.' + str(stage) + '.npy'

        np.save(file_name, self.z_dataset.batched_buffer)

        logging.critical('Save ZData to ' + file_name)

    def LoadZData(self, stage, id, params, FLAGS):

        checkpoint_dir = params['dirList']['checkpoint'] + FLAGS.checkpointStage
        file_name = checkpoint_dir + 'ZData.' + str(id) + '.s.' + str(stage) + '.npy'

        if not path.exists(file_name):
            return

        self.z_dataset.batched_buffer = np.load(file_name, allow_pickle=True)

        logging.critical('Load ZData from ' + file_name)
        self.z_dataset.init_state = False


    def saveState(self, epoch, id):
        return
        state_dict = {}

        state_dict['trainer_id'] = id
        state_dict['epoch'] = epoch

        # state_dict['z_model'] = self.z_model.state_dict()
        state_dict['p_model'] = self.p_model.state_dict()

        state_dict['z_optimizer'] = self.z_optimizer.state_dict()
        state_dict['p_optimizer'] = self.p_optimizer.state_dict()

        dir = 'trainer_buffer_2/'
        if not os.path.exists(dir):
            os.makedirs(dir)

        checkpointPath = dir + 'checkpoint-' + str(id) + '-e' + str(epoch) + ".pt"
        torch.save(state_dict, checkpointPath)


        np.save(dir+'zdata-' + str(id) + '-e' + str(epoch) + '.npy',self.z_dataset.batched_buffer)

        logging.critical('save trainer state' + '-' + str(id) + '-e' + str(epoch))

    def loadState(self, epoch, id):

        dir = 'trainer_buffer_2/'

        checkpoint = torch.load(dir + 'checkpoint-' + str(id) + '-e' + str(epoch) + ".pt", map_location=self.device)

        self.p_model.load_state_dict(checkpoint['p_model'])
        self.p_optimizer.load_state_dict(checkpoint['p_optimizer'])
        self.z_optimizer.load_state_dict(checkpoint['z_optimizer'])


        self.z_dataset.batched_buffer = np.load(dir + 'zdata-' + str(id) + '-e' + str(epoch) + '.npy', allow_pickle=True)
        logging.critical('load trainer state' + '-' + str(id) + '-e' + str(epoch))
        return True


    def __init__(self, params, z_model=None, z_criterion=None, z_optimizer=None, p_model=None, p_criterion=None, p_optimizer=None, p_train_loader=None, p_test_loader=None, p_error_func=None):

        self.device = params['device']

        self.step = params['globalStep']

        self.p_model = p_model
        self.p_criterion = p_criterion
        self.p_optimizer = p_optimizer
        self.p_train_loader = p_train_loader
        self.p_test_loader = p_test_loader
        self.p_error_func = p_error_func

        self.z_model = z_model
        self.z_criterion = z_criterion
        self.z_optimizer = z_optimizer
        self.z_dataset = self.BatchedDataset(len(self.p_train_loader), self.device)


        self.report_step = 500
        self.stage = 'c'

        self.z_updates = t_utility.GatherList(self.z_model, 'z_updates') + t_utility.GatherList(self.z_model, 'updates')
        self.p_updates = t_utility.GatherList(self.p_model, 'p_updates') + t_utility.GatherList(self.p_model, 'updates')


        self.z_parameters = t_utility.GatherDict(self.p_model, 'z_parameters')
        self.p_parameters = t_utility.GatherDict(self.z_model, 'p_parameters')

        self.count = 0

        self.freeze = False

    def run(self, epoch=0, P=True,Z=True, log=True):

        self.z_model.train()
        self.p_model.train()


        self.count += 1
        self.z_dataset.epoch()

        z_loss = 0
        p_loss = 0


        for batch_idx, (p_inputs, p_targets) in enumerate(self.p_train_loader):

            if Z == True:
                z_inputs, z_targets = self.z_dataset.batch(batch_idx)

                def z_closure():

                    self.z_optimizer.zero_grad()
                    z_outputs = self.z_model(z_inputs)

                    loss = self.z_criterion(z_outputs, z_targets)
                    loss.backward()
                    return loss

                z_loss = self.z_optimizer.step(z_closure)


                with torch.no_grad():
                    for update in self.z_updates:
                        update()


            if P == True:
                p_inputs = p_inputs.to(self.device)
                p_targets = p_targets.to(self.device)


                def p_closure():

                    self.p_optimizer.zero_grad()

                    p_outputs = self.p_model(p_inputs)

                    loss = self.p_criterion(p_outputs, p_targets)


                    # layer loss
                    layer_loss = sum(t_utility.GatherList(self.p_model, 'layer_loss'))

                    total_loss = layer_loss + loss

                    total_loss.backward()

                    return loss, layer_loss



                p_loss, regularization = self.p_optimizer.step(p_closure)

                with torch.no_grad():
                    for update in self.p_updates:
                        update()

            self.step += 1

        logging.info("Steps: %-5d ------------ Z : %f | P : %f | reg : %f " % (self.step, z_loss, p_loss, regularization))



    def evaluate(self, epoch=0, log=True):

        if self.p_test_loader is not None:

            error = t_utility.EvaluateErrors(self.p_test_loader, self.p_model, self.p_error_func).cpu().numpy()
            error_one_batch = t_utility.EvaluateErrors(self.p_test_loader, self.p_model, self.p_error_func, 1).cpu().numpy()

            name = str(self.p_error_func)
            logging.critical('P model evaluation | ' + name + ' ' + str(error) + ' ' + str(error_one_batch))

            if log == True and self.log_writer is not None:
                self.log_writer.add_scalar(self.log_space + 'test/P/' + name, error, epoch)

