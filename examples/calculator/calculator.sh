#!/bin/bash

echo make sure you copied the data folder of nn-mapping before you start this script

mkdir -p logs

echo train networks...

python ../../train.py projects/long-lstm10-b64.json
python ../../train.py projects/long-lstm30-b64.json
python ../../train.py projects/short-lstm10-b64.json
python ../../train.py projects/short-lstm30-b64.json

echo predict output

python ../../predict.py projects/long-lstm30-b64.json --input=input.json --output=output.json
