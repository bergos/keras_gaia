# Keras Gaia

Keras Gaia handles datasets and models for [Keras](https://keras.io/) in simple projects.
A simple abstraction is used in the dataset and model to allow easy interchanges.
The training of the network can be done with a command line tool using the project configuration.
The trained network can be used to make predication with a command line tool or and http service. 

## Requirements

The following dependencies are used:

- h5py
- ijson
- isodate
- keras

## Usage

The project settings are stored in a JSON file with the following structure:

- `label':
- `description`:
- `dataset`:
  - `trainingDataJson`:
  - `testDataJson`
- `model`:
  - `ioNamesJson`:
  - `topologyPython`:
- `training`:
  - `batchSize`:
  - `epochs`:
  - `shuffle`:
  - `lossLogFile`:
  - `testLogFile`:
  - `testLogInterval`:
  - `checkpointFile`:
  - `checkpointInterval`:

### train.py

Runs the training for the network based on the settings given as path to the project JSON file.
The following options are supported: 

- `base`: The base path for all files (optional)
- `resume`: Resume the training at the given epoch (optional)

### predict.py

Predicts the output for the given input based on the settings given as path to the project JSON file.
Input and output is read/written from/to JSON files.

- `base`: The base path for all files (optional)
- `input`: Path to the JSON file for the input data
- `output`: Path to the JSON file for the output data

### predict-http.py

Predicts the output for the given input based on the settings given as path to the project JSON file.
Input is read from the content of the POST request.
The input is expected as JSON string.
Output is written to the response as JSON string.

- `base`: The base path for all files (optional)
- `port`: Port for the HTTP server

## Example: Calculator

The package comes with a simple calculator example.
Use The JavaScript [nn-mapping](https://github.com/bergos/nn-mapping) package to generate example datasets.
Copy the `data` folder of nn-mapping to `examples/calculator/data`.
It contains two datasets (short and long).
The example defines two different models (lstm10 and lstm30).
The four projects use all combinations of the datasets and models.
Have a look at the `calculator.sh` bash file to see how the training is done and predictions are made.
