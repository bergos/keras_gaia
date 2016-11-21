import os.path as path
import utils


def load(config, options=None):
    label = None
    description = None
    io_names = None
    topology = None
    weights_hdf5 = None

    if options is None:
        options = {}

    if 'label' in config:
        label = config['label']

    if 'description' in config:
        description = config['description']

    if 'ioNamesJson' in config:
        filename = config['ioNamesJson']

        if not filename.startswith('/'):
            filename = path.join(options['base'], filename)

        io_names = utils.load_json(filename)

    if 'topologyPython' in config:
        filename = config['topologyPython']

        if not filename.startswith('/'):
            filename = path.join(options['base'], filename)

        topology = utils.load_python(filename)

    if 'weightsHdf5' in config:
        weights_hdf5 = config['weightsHdf5']

        if not weights_hdf5.startswith('/'):
            weights_hdf5 = path.join(options['base'], weights_hdf5)

    model = Model(label=label, description=description, io_names=io_names, topology=topology, weights_hdf5=weights_hdf5)

    if 'loadWeights' in options and options['loadWeights']:
        model.load_weights_hdf5()

    return model


class Model:
    def __init__(
            self,
            label=None,
            description=None,
            io_names=None,
            topology=None,
            weights_hdf5=None):

        self.label = label
        self.description = description
        self.io_names = io_names
        self.topology = topology
        self.weights_hdf5 = weights_hdf5

    def load_weights_hdf5(self):
        self.topology.load_weights(self.weights_hdf5)

    def save_weights_hdf5(self):
        self.topology.save_weights(self.weights_hdf5)
