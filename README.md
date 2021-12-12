# ExpertSearch with BERT Named Entity Recognition (Fall 2021 Course Project Update)
The code added to ExpertSearch in this project concerns the BERT NER model training, entity extraction, and batch extraction.

BERT NER is performed by fine-tuning a classification model based on the pre-trained DistilBERT version of Google's BERT model.

https://en.wikipedia.org/wiki/BERT_(language_model)

BERT is a natural language processing model that gives state of the art performance on a wide range of NLP tasks.

DistilBERT is a model that uses distillation techniques to reduce the size of BERT while keeping the large majority of its performance.

https://arxiv.org/abs/1910.01108

The training of this project's NER classification model leverages the HuggingFace transformers library.

The Transformers library provides quick and effective methods to train BERT derived models.

Once the model is trained, a prediction module is used to load the fine-tuned model and  make NER classifications on text inputs.

Unlike ExpertSearch's previous NER, which used a combination of regex and a NER toolkit from stanford, all of the NER is done in one extraction module,
using the fine-tuned model.

The extraction module breaks the faculty bio data into ngram tokens - strings of variable length, up to length six - and attempts to classify the text.

"The UIUC student"

=> becomes [The, The UIUC, The UIUC Student]

And each of these strings is fed to the BERT model to make a classification. 

Once the classifications are made, the text is written to a file, named after the entity classification it belongs to.

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

# Previous ExpertSearch README
## ExpertSearch

To run the code, run the following command from `ExpertSearch` (tested with Python2.7 on MacOS and Linux):

`gunicorn server:app -b 127.0.0.1:8095` 

The site should be available at http://localhost:8095/
