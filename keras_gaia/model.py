import os.path as path
import utils


def load(config, options=None):
    label = None
    description = None
    io_names = None
    topology = None

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

    return Model(label=label, description=description, io_names=io_names, topology=topology)


class Model:
    def __init__(
            self,
            label=None,
            description=None,
            io_names=None,
            topology=None):

        self.label = label
        self.description = description
        self.io_names = io_names
        self.topology = topology

    def load_weights_hdf5(self, filename):
        self.topology.load_weights(filename)

    def save_weights_hdf5(self, filename):
        self.topology.save_weights(filename)
