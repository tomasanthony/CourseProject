import numpy as np
from nltk import everygrams
import os, codecs
import bert_ner.bert_ner


def main(bio_dir, name_path):
    ner = bert_ner.bert_ner.BERTNer()
    entity_dict = {}
    for i in range(len(os.listdir(bio_dir)) - 1):
        with codecs.open(os.path.join(bio_dir, str(i) + '.txt'), 'r', encoding='utf-8', errors='ignore') as f:
            text = f.read()
        split_text = text.split()
        # Generate ngram tokens to then attempt to classify different lengths of text, up to length = 6 tokens
        ngrams = list(everygrams(split_text, max_len=6))
        tokens = np.array(list(map(' '.join, ngrams)))
        for token in tokens:
            text, entity = ner.classify_entity(token)
            if entity is not None:
                print(entity)
                if entity_dict.get(entity) is None:
                    entity_dict[entity] = []
                    entity_dict[entity].append(text)
                else:
                    entity_dict[entity].append(text)
        for key in entity_dict:
            write_path = f'{name_path}/{key}.txt'
            with open(write_path, 'w') as f:
                for text in entity_dict[key]:
                    f.write(text)
                    f.write('\n')


if __name__ == '__main__':
    bio_dir = '../data/compiled_bios/'
    name_path = '../data/'
    main(bio_dir, name_path)
