import os
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
import pathlib
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


class BERTNer:
    def __init__(self):
        self.recognition_threshold = .10
        bert_directory = str(pathlib.Path(__file__).parent / 'bert_ner_model')
        tokenizer_directory = str(pathlib.Path(__file__).parent / 'distilbert-base-uncased')
        self.tokenizer = DistilBertTokenizerFast.from_pretrained(tokenizer_directory)
        self.model = DistilBertForSequenceClassification.from_pretrained(bert_directory)

        corpora_path = str(pathlib.Path(__file__).parent.parent / 'data')

        corpora = pd.read_csv(os.path.join(corpora_path, "bert_corpora.csv"))
        y = corpora['Entity']
        label_encoder = LabelEncoder()
        label_encoder.fit_transform(y)
        self.code_label = dict(zip(range(len(label_encoder.classes_)), label_encoder.classes_))
        self.label_code = {value: key for key, value in self.code_label.items()}

    def predict(self, text_input):
        tokenized_inputs = self.tokenizer(text_input, return_tensors="pt")
        rankings = self.model(**tokenized_inputs)
        rankings_list = rankings.logits.softmax(dim=1).tolist()
        ner_prediction = np.argmax(rankings_list[0])

        entities = self.code_label

        return entities[ner_prediction], rankings_list[0][ner_prediction]

    def classify_entity(self, text):
        # If the entity prediction reaches the confidence threshold, classify the text as the entity
        entity, pred_score = self.predict(text)
        if pred_score < self.recognition_threshold:
            return None, None
        else:
            return text, entity




