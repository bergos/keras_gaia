import ijson.backends.yajl2_cffi as ijson
import numpy as np
import sys
import utils


def to_array_row(io_names, row, data):
    for index, io_name in enumerate(io_names):
        if io_name in row:
            data[index] = row[io_name]


def to_array(io_names, array, data, dim):
    if len(dim) == 1:
        to_array_row(io_names, array, data)
    else:
        for index, row, in enumerate(array):
            to_array(io_names, row, data[index], dim[1:])


def from_array_row(io_names, row):
    data = {}

    for index, io_name in enumerate(io_names):
        data[io_name] = float(row[index])

    return data


def from_array(io_names, array, dim):
    if len(dim) == 1:
        return from_array_row(io_names, array)
    else:
        data = []

        for index, row in enumerate(array):
            data.append(from_array(io_names, row, dim[1:]))

        return data


def get_data_dim(data, dim):
    if isinstance(data, list):
        dim.append(0)
        get_data_dim(data[0], dim)

    return dim


def get_max_dim(data, dim):
    if isinstance(data, list):
        dim[0] = max(dim[0], len(data))

    return dim


def get_item_dims(item, model, dims):
    for group in model.io_names:
        if group not in dims:
            dims[group] = get_data_dim(item[group], [])

        get_max_dim(item[group], dims[group])


def get_io_len(model, group):
    io_len = len(model.io_names[group])

    if group == 'input':
        shape = model.topology.input_shape[-1:]

        # model topology > model io?
        if len(shape) > 0 and shape[0] is not None:
            io_len = max(io_len, shape[0])

    return io_len


def load_batches_dims(filename, model, batch_size=0):
    batch_index = 0
    batch_dims = []
    item_count = 0
    index = 0
    progress = utils.ProgressText()

    with open(filename, 'r') as jsonfile:
        items = ijson.items(jsonfile, 'item')

        for index, item in enumerate(items):
            if batch_size > 0:
                batch_index = int(index / batch_size)

            # start of batch
            if batch_index >= len(batch_dims):
                batch_dims.append({})

            get_item_dims(item['data'], model, batch_dims[batch_index])

            item_count += 1

            # end of batch
            if batch_size == item_count:
                for group in model.io_names:
                    batch_dims[batch_index][group] = \
                        [item_count] + batch_dims[batch_index][group] + [get_io_len(model, group)]

                item_count = 0

            if ((index + 1) % 100) == 0:
                progress.text('count items: ' + str(index + 1))

        progress.text('count items: ' + str(index + 1))
        sys.stdout.write('\n')

        if item_count != 0:
            for group in model.io_names:
                batch_dims[batch_index][group] = \
                    [item_count] + batch_dims[batch_index][group] + [get_io_len(model, group)]

    return batch_dims


def load_batches(filename, model, batch_size=0):
    batch_index = 0
    batches = []
    item_count = 0
    index = 0
    group = ''
    progress = utils.ProgressText()

    batch_dims = load_batches_dims(filename, model, batch_size)

    with open(filename, 'r') as jsonfile:
        items = ijson.items(jsonfile, 'item')

        for index, item in enumerate(items):
            if batch_size > 0:
                batch_index = int(index / batch_size)

            # start of batch
            if batch_index >= len(batches):
                batches.append({'labels': []})

                for group in model.io_names:
                    batches[batch_index][group] = np.zeros(batch_dims[batch_index][group], dtype=np.float32)

            for group in model.io_names:
                to_array(
                    model.io_names[group],
                    item['data'][group],
                    batches[batch_index][group][item_count],
                    batch_dims[batch_index][group][1:]
                )

            if 'label' in item:
                batches[batch_index]['labels'].append(item['label'])
            else:
                batches[batch_index]['labels'].append('')

            item_count += 1

            # end of batch
            if batch_size == item_count:
                item_count = 0

            if ((index + 1) % 100) == 0:
                progress.text('import items: ' + str(index + 1) + '/' + str(batch_dims[batch_index][group][0]))

        progress.text('import items: ' + str(index + 1) + '/' + str(batch_dims[batch_index][group][0]))
        sys.stdout.write('\n')

    return batches


def map_group_item(model, group, data):
    dims = [1] + [len(data)] + [get_io_len(model, group)]

    array = np.zeros(dims, dtype=np.float32)

    to_array(model.io_names[group], [data], array, dims)

    return array


def map_array_group_item(model, group, array):
    return from_array(model.io_names[group], array, array.shape)[0]


def load(filename, model, batch_size=0):
    batches = load_batches(filename, model, batch_size)

    return batches
