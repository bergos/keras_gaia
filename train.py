import argparse
import keras_gaia.project as project_utils

parser = argparse.ArgumentParser(description='Train neural network based on a project file')
parser.add_argument('--base', help='base path for relative pathes in project file')
parser.add_argument('--resume', type=int, help='resume at the given epoch')
parser.add_argument('projectFile', help='project file')

args = parser.parse_args()

project = project_utils.load_json(args.projectFile, {
    'base': args.base,
    'loadTrainingData': True,
    'loadTestData': True
})

if args.resume is not None:
    project.train_resume(args.resume)
else:
    project.train()
