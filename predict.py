import argparse
import keras_gaia.project as project_utils
import keras_gaia.utils as utils

parser = argparse.ArgumentParser(description='Use a neural network project to predict data')
parser.add_argument('--base', help='base path for relative pathes in project file')
parser.add_argument('--input', help='input data file')
parser.add_argument('--output', help='output data file')
parser.add_argument('projectFile', help='project file')

args = parser.parse_args()

project = project_utils.load_json(args.projectFile, {
    'base': args.base,
    'loadWeights': True
})

input_data = utils.load_json(args.input)

output_data = project.predict(input_data)

utils.save_json(args.output, output_data)
