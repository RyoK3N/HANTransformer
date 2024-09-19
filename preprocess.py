# preprocess.py

import os
import json
import torch
import spacy
import numpy as np
from sklearn.datasets import fetch_20newsgroups
from collections import Counter
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

# Install spaCy model if not already installed
try:
    nlp = spacy.load("en_core_web_sm")
except:
    import subprocess
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm")

# Parameters
MAX_SENTENCES = 30  # Maximum number of sentences per document
MAX_WORDS = 50      # Maximum number of words per sentence
MAX_RULES = 3       # Maximum rules per word
MIN_WORD_FREQ = 5   # Minimum frequency to include a word in the vocabulary

# Paths
DATA_DIR = "data"
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

def tokenize_text(text):
    """
    Tokenizes text into sentences and words using spaCy.
    """
    doc = nlp(text)
    sentences = []
    for sent in doc.sents:
        words = [token.text.lower() for token in sent if not token.is_punct and not token.is_space]
        if words:
            sentences.append(words)
    return sentences

def assign_rules(pos_tags):
    """
    Assigns dummy rules based on POS tags.
    """
    rules = []
    for pos in pos_tags:
        word_rules = []
        if pos.startswith('N'):
            word_rules.append('rule1')
        if pos.startswith('V'):
            word_rules.append('rule2')
        if not word_rules:
            word_rules.append('rule3')
        # Ensure max_rules_per_word
        word_rules = word_rules[:MAX_RULES]
        rules.append(word_rules)
    return rules

def build_vocab(tokenized_texts, min_freq=MIN_WORD_FREQ):
    """
    Builds a vocabulary dictionary mapping tokens to indices.
    """
    counter = Counter()
    for doc in tokenized_texts:
        for sentence in doc:
            counter.update(sentence)
    vocab = {'<PAD>':0, '<UNK>':1}
    for word, freq in counter.items():
        if freq >= min_freq:
            vocab[word] = len(vocab)
    return vocab

def build_pos_vocab(tokenized_texts):
    """
    Builds a vocabulary for POS tags.
    """
    pos_counter = Counter()
    for doc in tokenized_texts:
        for sentence in doc:
            doc_spacy = nlp(" ".join(sentence))
            pos_counter.update([token.pos_ for token in doc_spacy])
    pos_vocab = {'<PAD>':0, '<UNK>':1}
    for pos in pos_counter:
        pos_vocab[pos] = len(pos_vocab)
    return pos_vocab

def build_rule_vocab(tokenized_texts):
    """
    Builds a vocabulary for rules.
    """
    # Since rules are predefined as rule1, rule2, rule3
    rule_vocab = {'<PAD>':0, 'rule1':1, 'rule2':2, 'rule3':3}
    return rule_vocab

def encode_text(tokenized_texts, vocab, max_sentences=MAX_SENTENCES, max_words=MAX_WORDS):
    """
    Encodes tokenized texts into IDs and pads/truncates them.
    """
    encoded_texts = []
    for doc in tokenized_texts:
        encoded_doc = []
        for sent in doc[:max_sentences]:
            encoded_sent = [vocab.get(word, vocab['<UNK>']) for word in sent[:max_words]]
            if len(encoded_sent) < max_words:
                encoded_sent += [vocab['<PAD>']] * (max_words - len(encoded_sent))
            encoded_doc.append(encoded_sent)
        if len(encoded_doc) < max_sentences:
            pad_sentence = [vocab['<PAD>']] * max_words
            encoded_doc += [pad_sentence] * (max_sentences - len(encoded_doc))
        encoded_texts.append(encoded_doc)
    return encoded_texts

def encode_pos(tokenized_texts, pos_vocab, max_sentences=MAX_SENTENCES, max_words=MAX_WORDS):
    """
    Encodes POS tags into IDs and pads/truncates them.
    """
    encoded_pos = []
    for doc in tokenized_texts:
        encoded_doc = []
        for sent in doc[:max_sentences]:
            doc_spacy = nlp(" ".join(sent[:max_words]))
            pos_tags = [token.pos_ for token in doc_spacy]
            encoded_sent = [pos_vocab.get(pos, pos_vocab['<UNK>']) for pos in pos_tags]
            if len(encoded_sent) < max_words:
                encoded_sent += [pos_vocab['<PAD>']] * (max_words - len(encoded_sent))
            else:
                encoded_sent = encoded_sent[:max_words]
            encoded_doc.append(encoded_sent)
        if len(encoded_doc) < max_sentences:
            pad_sentence = [pos_vocab['<PAD>']] * max_words
            encoded_doc += [pad_sentence] * (max_sentences - len(encoded_doc))
        encoded_pos.append(encoded_doc)
    return encoded_pos

def encode_rules(tokenized_texts, rule_vocab, max_sentences=MAX_SENTENCES, max_words=MAX_WORDS, max_rules=MAX_RULES):
    """
    Encodes rules into IDs and pads/truncates them.
    """
    encoded_rules = []
    for doc in tokenized_texts:
        encoded_doc = []
        for sent in doc[:max_sentences]:
            doc_spacy = nlp(" ".join(sent[:max_words]))
            pos_tags = [token.pos_ for token in doc_spacy]
            word_rules = assign_rules(pos_tags)
            encoded_sent = []
            for rules in word_rules[:max_words]:
                encoded_rule = [rule_vocab.get(rule, rule_vocab['<PAD>']) for rule in rules]
                if len(encoded_rule) < max_rules:
                    encoded_rule += [rule_vocab['<PAD>']] * (max_rules - len(encoded_rule))
                encoded_sent.append(encoded_rule)
            if len(encoded_sent) < max_words:
                pad_rule = [rule_vocab['<PAD>']] * max_rules
                encoded_sent += [pad_rule] * (max_words - len(encoded_sent))
            encoded_doc.append(encoded_sent)
        if len(encoded_doc) < max_sentences:
            pad_sentence = [[rule_vocab['<PAD>']] * max_rules for _ in range(max_words)]
            encoded_doc += [pad_sentence] * (max_sentences - len(encoded_doc))
        encoded_rules.append(encoded_doc)
    return encoded_rules

def create_attention_mask(encoded_texts, max_sentences=MAX_SENTENCES, max_words=MAX_WORDS):
    """
    Creates attention masks for the input_ids.
    1 indicates real tokens, 0 indicates padding.
    """
    attention_masks = []
    for doc in encoded_texts:
        doc_mask = []
        for sent in doc[:max_sentences]:
            sent_mask = [1 if word_id != 0 else 0 for word_id in sent[:max_words]]
            sent_mask += [0] * (max_words - len(sent_mask))
            doc_mask.append(sent_mask)
        if len(doc_mask) < max_sentences:
            pad_mask = [0] * max_words
            doc_mask += [pad_mask] * (max_sentences - len(doc_mask))
        attention_masks.append(doc_mask)
    return attention_masks

def create_sentence_masks(encoded_texts, max_sentences=MAX_SENTENCES, max_words=MAX_WORDS):
    """
    Creates sentence masks for the documents.
    1 indicates real sentences, 0 indicates padding.
    """
    sentence_masks = []
    for doc in encoded_texts:
        mask = [1 if any(word_id != 0 for word_id in sent[:max_words]) else 0 for sent in doc[:max_sentences]]
        mask += [0] * (max_sentences - len(mask))
        sentence_masks.append(mask)
    return sentence_masks

def encode_labels(targets):
    """
    Encodes labels into integers.
    """
    label_to_id = {label: idx for idx, label in enumerate(sorted(set(targets)))}
    encoded_labels = [label_to_id[label] for label in targets]
    return encoded_labels, label_to_id

def save_json(obj, path):
    """
    Save JSON after ensuring keys are Python `int`, not `int64`.
    """
    def convert(obj):
        if isinstance(obj, np.int64):
            return int(obj)
        if isinstance(obj, dict):
            return {convert(k): convert(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [convert(i) for i in obj]
        return obj
    
    with open(path, 'w') as f:
        json.dump(convert(obj), f)

def parallel_tokenize_texts(texts):
    """
    Parallelize text tokenization using multiprocessing.
    """
    with Pool(cpu_count()) as pool:
        tokenized_texts = list(tqdm(pool.imap(tokenize_text, texts), total=len(texts), desc="Tokenizing"))
    return tokenized_texts

def main():
    # Load dataset
    print("Loading 20 Newsgroups dataset...")
    newsgroups = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))
    texts = newsgroups.data
    targets = newsgroups.target
    target_names = newsgroups.target_names

    # Tokenize texts using multiprocessing for faster execution
    print("Tokenizing texts into sentences and words...")
    tokenized_texts = parallel_tokenize_texts(texts)

    # Build vocabularies
    print("Building vocabularies...")
    word_vocab = build_vocab(tokenized_texts)
    pos_vocab = build_pos_vocab(tokenized_texts)
    rule_vocab = build_rule_vocab(tokenized_texts)

    print(f"Word Vocabulary Size: {len(word_vocab)}")
    print(f"POS Vocabulary Size: {len(pos_vocab)}")
    print(f"Rule Vocabulary Size: {len(rule_vocab)}")

    # Encode texts
    print("Encoding texts...")
    encoded_texts = encode_text(tokenized_texts, word_vocab)
    encoded_pos = encode_pos(tokenized_texts, pos_vocab)
    encoded_rules = encode_rules(tokenized_texts, rule_vocab)
    attention_masks = create_attention_mask(encoded_texts)
    sentence_masks = create_sentence_masks(encoded_texts)
    encoded_labels, label_to_id = encode_labels(targets)

    # Split into train and test
    from sklearn.model_selection import train_test_split
    print("Splitting data into train and test sets...")
    train_texts, test_texts, train_pos, test_pos, train_rules, test_rules, train_masks, test_masks, train_sentence_masks, test_sentence_masks, train_labels, test_labels = train_test_split(
        encoded_texts,
        encoded_pos,
        encoded_rules,
        attention_masks,
        sentence_masks,
        encoded_labels,
        test_size=0.2,
        random_state=42
    )

    # Save processed data
    print("Saving processed data...")
    processed_data = {
        'train': {
            'input_ids': train_texts,
            'pos_tags': train_pos,
            'rules': train_rules,
            'attention_mask': train_masks,
            'sentence_masks': train_sentence_masks,
            'labels': train_labels
        },
        'test': {
            'input_ids': test_texts,
            'pos_tags': test_pos,
            'rules': test_rules,
            'attention_mask': test_masks,
            'sentence_masks': test_sentence_masks,
            'labels': test_labels
        },
        'vocab': {
            'word_vocab': word_vocab,
            'pos_vocab': pos_vocab,
            'rule_vocab': rule_vocab,
            'label_to_id': label_to_id,
            'id_to_label': {v: k for k, v in label_to_id.items()},
            'target_names': target_names
        }
    }

    save_json(processed_data, os.path.join(DATA_DIR, 'processed_data.json'))
    
    print("Preprocessing completed successfully.")

if __name__ == "__main__":
    main()
