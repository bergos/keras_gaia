from training import CsvLossLogger
from training import Checkpoint
from training import TestCallback
import data as data_utils
import dataset as dataset_utils
import model as model_utils
import os
import sys
import training as training_utils
import utils


def load(config, options=None):
    label = None
    description = None
    dataset = None
    model = None
    training = None

    if options is None:
        options = {}

    if 'base' not in options or options['base'] is None:
        options['base'] = os.getcwd()

    if not options['base'].startswith('/'):
        options['base'] = os.path.join(os.getcwd(), options['base'])

    if 'label' in config:
        label = config['label']

    if 'description' in config:
        description = config['description']

    if 'model' in config:
        model = model_utils.load(config['model'], options)

    if 'training' in config:
        training = training_utils.load(config['training'], options)

    if 'dataset' in config:
        dataset = dataset_utils.load(config['dataset'], model, training, options)

    return Project(
        label=label,
        description=description,
        dataset=dataset,
        model=model,
        training=training,
        options=options)


def load_json(filename, options=None):
    config = utils.load_json(filename)

    return load(config, options)


class Project:
    def __init__(
            self,
            label=None,
            description=None,
            dataset=None,
            model=None,
            training=None,
            options=None):

        self.label = label
        self.description = description
        self.dataset = dataset
        self.model = model
        self.training = training
        self.options = options
        self.training_callbacks = []

    def train_callbacks(self):
        callbacks = []

        if self.training.loss_log_file is not None:
            filename = self.training.loss_log_file

            if 'base' in self.options and not filename.startswith('/'):
                filename = os.path.join(self.options['base'], filename)

            callbacks += [CsvLossLogger(
                filename,
                interval=self.training.loss_log_interval,
                epoch=self.training.start_epoch
            )]

        if self.training.checkpoint_file is not None:
            filename = self.training.checkpoint_file

            if 'base' in self.options and not filename.startswith('/'):
                filename = os.path.join(self.options['base'], filename)

            callbacks += [Checkpoint(
                filename,
                self.model,
                interval=self.training.checkpoint_interval,
                epoch=self.training.start_epoch
            )]

        if self.training.test_log_file is not None:
            filename = self.training.test_log_file

            if 'base' in self.options and not filename.startswith('/'):
                filename = os.path.join(self.options['base'], filename)

            callbacks += [TestCallback(
                filename,
                self,
                interval=self.training.test_log_interval,
                epoch=self.training.start_epoch
            )]

        return callbacks

    def batch_generator(self):
        while True:
            for batches in self.dataset.training_data:
                yield (batches['input'], batches['output'])

    def train(self, callbacks=None):
        if callbacks is None:
            callbacks = []

        callbacks += self.train_callbacks()

        if self.training.varying_batch_size:
            # TODO: implement shuffle

            samples_per_epoch = sum(map(lambda x: len(x['input']), self.dataset.training_data))

            self.model.topology.fit_generator(
                generator=self.batch_generator(),
                samples_per_epoch=samples_per_epoch,
                nb_epoch=self.training.end_epoch - self.training.start_epoch,
                callbacks=callbacks)
        else:
            self.model.topology.fit(
                self.dataset.training_data['input'],
                self.dataset.training_data['output'],
                batch_size=self.training.batch_size,
                nb_epoch=self.training.end_epoch - self.training.start_epoch,
                shuffle=self.training.shuffle,
                callbacks=callbacks)

        self.model.save_weights_hdf5()

    def train_resume(self, epoch, callbacks=None):
        filename = self.training.checkpoint_file.format(epoch=epoch)

        self.model.topology.load_weights(filename)
        self.training.start_epoch = epoch + 1

        sys.stdout.write('resume epoch ' + str(self.training.start_epoch) + '\n')

        self.train(callbacks)

    def predict(self, input_data):
        input_array = data_utils.map_group_item(self.model, 'input', input_data)

        output_array = self.model.topology.predict(input_array)

        return data_utils.map_array_group_item(self.model, 'output', output_array)
