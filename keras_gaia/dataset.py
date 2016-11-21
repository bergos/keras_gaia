import data as data_utils
import os


def load(config, model, training, options=None):
    label = None
    description = None
    training_data = None
    test_data = None

    if options is None:
        options = {}

    if 'label' in config:
        label = config['label']

    if 'description' in config:
        description = config['description']

    if 'loadTrainingData' in options and options['loadTrainingData'] and 'trainingDataJson' in config:
        filename = config['trainingDataJson']

        if 'base' in options and not filename.startswith('/'):
            filename = os.path.join(options['base'], filename)

        if training.varying_batch_size:
            training_data = data_utils.load(filename, model, training.batch_size)
        else:
            training_data = data_utils.load(filename, model)[0]

    if 'loadTestData' in options and options['loadTestData'] and 'testDataJson' in config:
        filename = config['testDataJson']

        if 'base' in options and not filename.startswith('/'):
            filename = os.path.join(options['base'], filename)

        test_data = data_utils.load(filename, model)[0]

    return Dataset(label=label, description=description, training_data=training_data, test_data=test_data)


class Dataset:
    def __init__(
            self,
            label=None,
            description=None,
            training_data=None,
            test_data=None):

        self.label = label
        self.description = description
        self.training_data = training_data
        self.test_data = test_data
