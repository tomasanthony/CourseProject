from collections import Counter
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
import os
import pandas as pd
import pathlib
import pprint
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import torch
from transformers import DistilBertTokenizerFast
from transformers import DistilBertForSequenceClassification, TrainingArguments, Trainer

bert_directory = str(pathlib.Path(__file__).parent / 'distilbert-base-uncased')

if not os.path.isdir(bert_directory):
    bert_directory = 'distilbert-base-uncased'

corpora_path = str(pathlib.Path(__file__).parent.parent / 'data')

corpora = pd.read_csv(os.path.join(corpora_path, "bert_corpora.csv"))

print(corpora.Entity.value_counts())

x, y = corpora['Text'], corpora['Entity']

label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(y)
entity_mapping = dict(zip(range(len(label_encoder.classes_)), label_encoder.classes_))

# ros_strategy = {}
rus_strategy = {0: 1000, 1: 1000, 2: 1000, 3: 1000, 4: 1000, 5: 1000}

print('Dataset pre-resampling shape %s' % Counter(labels))

# Naive class rebalancing with oversampling, undersampling, or both.
# Initial sampling strategy must have the x values reshaped
# Secondary sampling must take in the result of the initial sampling as arguments
# oversampler = RandomOverSampler(sampling_strategy=ros_strategy, random_state=64)
undersampler = RandomUnderSampler(sampling_strategy=rus_strategy, random_state=64)

# x_ros, y_ros = oversampler.fit_resample(x.values.reshape((-1, 1)), labels)
# x_final, y_final = undersampler.fit_resample((x_ros, y_ros))

x_final, y_final = undersampler.fit_resample(x.values.reshape((-1, 1)), labels)

print('Dataset post_resampling shape %s' % Counter(y_final))

df = pd.DataFrame(x_final)
train_text, test_text, train_labels, test_labels = train_test_split(x_final.flatten(), y_final, random_state=32,
                                                                    test_size=250, stratify=y_final)

model = DistilBertForSequenceClassification.from_pretrained(bert_directory, num_labels=len(y.unique()))
tokenizer = DistilBertTokenizerFast.from_pretrained(bert_directory)
model.save_pretrained(bert_directory)
tokenizer.save_pretrained(bert_directory)

train_encodings = tokenizer(train_text.tolist(), truncation=True, padding=True)
eval_encodings = tokenizer(test_text.tolist(), truncation=True, padding=True)


class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels=None):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]).long() for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx]).long()
        return item

    def __len__(self):
        return len(self.labels)


train_dataset = Dataset(train_encodings, labels=train_labels)
eval_dataset = Dataset(eval_encodings, labels=test_labels)

print(f'Size of the training dataset: {len(train_dataset)}')
print(f'Size of the testing dataset: {len(eval_dataset)}')

training_args = TrainingArguments(
    output_dir='./model-training',
    num_train_epochs=10,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=64,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
)


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics
)

trainer.train()
trainer.save_model('./bert_ner_model')
evaluation = trainer.evaluate()

print(f'Performance metrics')
pprint = pprint.PrettyPrinter(indent=4)
pprint.pprint(evaluation)
