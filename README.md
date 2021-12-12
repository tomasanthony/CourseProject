# ExpertSearch with BERT Named Entity Recognition (Fall 2021 Course Project Update)
In order to execute named entity recognition (NER) with the BERT classification model:

## Install requirements

From the project root directory:

`python -m pip install -r bert-ner/train_requirements.txt`

## Execute train.py

Run the `train.py` file. Training can take an extensive amount of time. For testing purposes, the number of training epochs,
dataset size (through the alteration of undersampling arguments in `train.py`, and warmup steps can all be reduced.

## Run the extractor

Run the `extract_bert.py` file. Text files will be written to the data directory.

The file name pattern will be <EntityName>.txt, with the text of the entities recognized on each line of the file.
e.g 
File - PERSON.txt
Contents - Tomas Anthony
           John Smith
           Jane Doe

# Demonstration
Link:

https://uillinoisedu-my.sharepoint.com/:v:/g/personal/tomasaa2_illinois_edu/EUwiwWKyskJLrwxgM4S-mQEBqGD3A2o5hJdX5pbYV8KwHA?e=QEiXa8

# ExpertSearch

To run the code, run the following command from `ExpertSearch` (tested with Python2.7 on MacOS and Linux):

`gunicorn server:app -b 127.0.0.1:8095` 

The site should be available at http://localhost:8095/
