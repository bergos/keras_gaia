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

- `label': Name of the project
- `description`: Description of the project
- `weightsHdf5`: Path to the HDF5 weights file.
  File will be written after the training and read before prediction.
- `dataset`: Dataset options
  - `trainingDataJson`: Path to the JSON training data file
  - `testDataJson` Path to the JSON test data file
- `model`: Model options
  - `ioNamesJson`: Path to the JSON I/O names definition file
  - `topologyPython`: Path to the python code that creates the Keras model.  
    The code must contain a `create` function which returns the model.  
- `training`: Training options
  - `batchSize`: Batch size
  - `epochs`: Number of epochs
  - `shuffle`: Shuffle the dataset on each epoch
  - `lossLogFile`: Path to the CSV loss log file 
  - `testLogFile`: Path to the CSV test log file
  - `testLogInterval`: The epoch interval for the tests
  - `checkpointFile`: File pattern to dump the weights during the training.
    Use Python string templates for epoch (e.g. {epoch:06d}).
  - `checkpointInterval`: The epoch interval for the weights dumps

### Training

The training step generates the weights for the model.
This can be done from the command line.
The following command runs the training for a project:

    python train.py [options] <projectFile>

- `projectFile`: Path to the JSON project file
- `base`: The base path for all files (optional)
- `resume`: Resume the training at the given epoch (optional)

### Prediction

Predictions can be made based on existing weights.
This can be done from the command line.
The input and output is read and written to JSON files. 
The following command runs a prediction for a project:

    python predict.py [options] --input=<inputFile> --output=<outputFile> <projectFile>

- `projectFile`: Path to the JSON project file
- `input`: Path to the JSON input file
- `output`: Path to the JSON output file
- `base`: The base path for all files (optional)

### Prediction HTTP server

It's also possible to use a HTTP interface for the predictions.
The input must be sent as JSON string with a POST request to the endpoint URL.
For example if port 8080 is used `http://localhost:8080/`.
The output is returned to the client as JSON string. 
The following command starts the HTTP prediction server for a project:

    python predict-http.py [options] --port=<port> <projectFile>

- `projectFile`: Path to the JSON project file
- `port`: Port for the HTTP server
- `base`: The base path for all files (optional)

## Example: Calculator

The package comes with a simple calculator example.
Use The JavaScript [nn-mapping](https://github.com/bergos/nn-mapping) package to generate the example datasets.
Copy the `data` folder in the examples folder of nn-mapping to `examples/calculator/data`.
It contains two datasets (short and long).
The example defines two different models (lstm10 and lstm30).
The four projects use all combinations of the datasets and models.
The `calculator.sh` bash file runs the training for all projects and makes a prediction.
