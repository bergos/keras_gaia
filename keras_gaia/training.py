from datetime import datetime
from isodate import datetime_isoformat
from keras.callbacks import Callback
import os.path as path


def load(config, options):
    batch_size = 32
    end_epoch = 10
    shuffle = False
    varying_batch_size = False
    loss_log_file = None
    loss_log_interval = 1
    test_log_file = None
    test_log_interval = 1
    checkpoint_file = None
    checkpoint_interval = 10

    if 'batchSize' in config:
        batch_size = config['batchSize']

    if 'epochs' in config:
        end_epoch = config['epochs']

    if 'shuffle' in config:
        shuffle = config['shuffle']

    if 'varyingBatchSize' in config:
        varying_batch_size = config['varyingBatchSize']

    if 'lossLogFile' in config:
        loss_log_file = config['lossLogFile']

        if not loss_log_file.startswith('/'):
            loss_log_file = path.join(options['base'], loss_log_file)

    if 'lossLogInterval' in config:
        loss_log_interval = config['lossLogInterval']

    if 'testLogFile' in config:
        test_log_file = config['testLogFile']

        if not test_log_file.startswith('/'):
            test_log_file = path.join(options['base'], test_log_file)

    if 'testLogInterval' in config:
        test_log_interval = config['testLogInterval']

    if 'checkpointFile' in config:
        checkpoint_file = config['checkpointFile']

        if not checkpoint_file.startswith('/'):
            checkpoint_file = path.join(options['base'], checkpoint_file)

    if 'checkpointInterval' in config:
        checkpoint_interval = config['checkpointInterval']

    return Training(
        batch_size=batch_size,
        end_epoch=end_epoch,
        shuffle=shuffle,
        varying_batch_size=varying_batch_size,
        loss_log_file=loss_log_file,
        loss_log_interval=loss_log_interval,
        test_log_file=test_log_file,
        test_log_interval=test_log_interval,
        checkpoint_file=checkpoint_file,
        checkpoint_interval=checkpoint_interval
    )


class Training:
    def __init__(
            self,
            batch_size=32,
            start_epoch=0,
            end_epoch=10,
            shuffle=False,
            varying_batch_size=False,
            loss_log_file=None,
            loss_log_interval=1,
            test_log_file=None,
            test_log_interval=1,
            checkpoint_file=None,
            checkpoint_interval=10):

        self.batch_size = batch_size
        self.start_epoch = start_epoch
        self.end_epoch = end_epoch
        self.shuffle = shuffle
        self.varying_batch_size = varying_batch_size
        self.loss_log_file = loss_log_file
        self.loss_log_interval = loss_log_interval
        self.test_log_file = test_log_file
        self.test_log_interval = test_log_interval
        self.checkpoint_file = checkpoint_file
        self.checkpoint_interval = checkpoint_interval


class Checkpoint(Callback):
    def __init__(self, filename, model, interval=1, epoch=0):
        self.filename = filename
        self.gaia_model = model
        self.interval = interval
        self.epoch = epoch

    def on_epoch_end(self, epoch, logs):
        if (self.epoch % self.interval) == 0:
            self.gaia_model.topology.save_weights(self.filename.format(epoch=self.epoch))

        self.epoch += 1


class CsvLossLogger(Callback):
    def __init__(self, filename, interval=1, epoch=0):
        self.filename = filename
        self.interval = interval
        self.epoch = epoch

        if epoch == 0 and path.isfile(self.filename):
            with open(self.filename, 'rw+') as log_file:
                log_file.truncate()

        with open(self.filename, 'a') as log_file:
            log_file.write('epoch\tbatch\ttimestamp\tloss\n')

    def on_batch_end(self, batch, logs):
        if (self.epoch % self.interval) == 0:
            with open(self.filename, 'a') as log_file:
                log_file.write('{0:d}\t{1:d}\t{2:s}\t{3:s}\n'.format(
                    self.epoch, batch, datetime_isoformat(datetime.now()), logs['loss']))

    def on_epoch_end(self, epoch, logs):
        self.epoch += 1


class TestCallback(Callback):
    def __init__(self, filename, project, interval=1, epoch=0):
        self.filename = filename
        self.project = project
        self.interval = interval
        self.epoch = epoch

        if epoch == 0 and path.isfile(self.filename):
            with open(self.filename, 'rw+') as log_file:
                log_file.truncate()

        with open(self.filename, 'a') as log_file:
            log_file.write('epoch\tbatch\ttimestamp\tlabel\texpected\tpredicted\n')

    def on_epoch_end(self, epoch, logs):
        self.epoch += 1

        if (self.epoch % self.interval) == 0:
            predictions = self.project.model.topology.predict(self.project.dataset.test_data['input'])
            timestamp = datetime_isoformat(datetime.now())

            with open(self.filename, 'a') as log_file:
                for i in range(0, len(predictions)):
                    label = self.project.dataset.test_data['labels'][i]
                    expected = self.project.dataset.test_data['output'][i][0]
                    predicted = predictions[i][0]

                    log_file.write('{0:d}\t\t{1:s}\t{2:s}\t{3:16.13f}\t{4:16.13f}\n'.format(
                        self.epoch, timestamp, label, expected, predicted))
