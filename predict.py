# predict.py

import os
import json
import torch
import torch.nn as nn
from model import get_model  # Ensure model.py is in the same directory
import spacy
import numpy as np
from tqdm import tqdm

# Set random seeds for reproducibility
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    import random
    random.seed(seed)

set_seed()

# Parameters
DATA_DIR = "data"
PROCESSED_DATA_PATH = os.path.join(DATA_DIR, 'processed_data.json')
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_RULES_PER_WORD = 3  # Should match preprocessing
MAX_SENTENCES = 30
MAX_WORDS = 50

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except:
    import subprocess
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm")

def load_vocab():
    """
    Loads vocabularies and label mappings from the processed_data.json file.
    """
    with open(PROCESSED_DATA_PATH, 'r') as f:
        data = json.load(f)
    vocab = data['vocab']
    target_names = vocab['target_names']
    word_vocab = vocab['word_vocab']
    pos_vocab = vocab['pos_vocab']
    rule_vocab = vocab['rule_vocab']
    return word_vocab, pos_vocab, rule_vocab, target_names

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
        word_rules = word_rules[:MAX_RULES_PER_WORD]
        rules.append(word_rules)
    return rules

def preprocess_text(text, word_vocab, pos_vocab, rule_vocab):
    """
    Preprocesses the input text:
    - Tokenizes into sentences and words
    - Assigns POS tags
    - Assigns rules
    - Encodes using vocabularies
    - Pads/truncates to fixed sizes
    Returns encoded input tensors.
    """
    # Tokenize text into sentences and words
    doc = nlp(text)
    sentences = []
    for sent in doc.sents:
        words = [token.text.lower() for token in sent if not token.is_punct and not token.is_space]
        if words:
            sentences.append(words)
    # Limit number of sentences
    if len(sentences) > MAX_SENTENCES:
        sentences = sentences[:MAX_SENTENCES]
    else:
        # Pad with empty sentences
        while len(sentences) < MAX_SENTENCES:
            sentences.append([])
    
    encoded_input_ids = []
    encoded_pos_tags = []
    encoded_rules = []
    attention_masks = []
    sentence_masks = []
    
    for sent in sentences:
        # Limit number of words
        if len(sent) > MAX_WORDS:
            sent = sent[:MAX_WORDS]
            pad_length = 0
        else:
            pad_length = MAX_WORDS - len(sent)
        
        # Encode words
        encoded_sent = [word_vocab.get(word, word_vocab.get('<UNK>')) for word in sent]
        # Pad words
        if pad_length > 0:
            encoded_sent += [word_vocab.get('<PAD>')] * pad_length
        encoded_input_ids.append(encoded_sent)
        
        # POS tagging
        doc_sent = nlp(" ".join(sent[:MAX_WORDS]))
        pos_tags = [token.pos_ for token in doc_sent]
        # Encode POS tags
        encoded_pos = [pos_vocab.get(pos, pos_vocab.get('<UNK>')) for pos in pos_tags]
        # Pad POS tags
        if len(encoded_pos) < MAX_WORDS:
            encoded_pos += [pos_vocab.get('<PAD>')] * (MAX_WORDS - len(encoded_pos))
        else:
            encoded_pos = encoded_pos[:MAX_WORDS]
        encoded_pos_tags.append(encoded_pos)
        
        # Assign rules
        word_rules = assign_rules(pos_tags)
        # Encode rules
        encoded_rule_sent = []
        for rules in word_rules[:MAX_WORDS]:
            encoded_rule = [rule_vocab.get(rule, rule_vocab.get('<PAD>')) for rule in rules]
            # Pad rules
            if len(encoded_rule) < MAX_RULES_PER_WORD:
                encoded_rule += [rule_vocab.get('<PAD>')] * (MAX_RULES_PER_WORD - len(encoded_rule))
            else:
                encoded_rule = encoded_rule[:MAX_RULES_PER_WORD]
            encoded_rule_sent.append(encoded_rule)
        # Pad rules for words
        if len(encoded_rule_sent) < MAX_WORDS:
            pad_rule = [rule_vocab.get('<PAD>')] * MAX_RULES_PER_WORD
            encoded_rule_sent += [pad_rule] * (MAX_WORDS - len(encoded_rule_sent))
        else:
            encoded_rule_sent = encoded_rule_sent[:MAX_WORDS]
        encoded_rules.append(encoded_rule_sent)
        
        # Attention mask for words
        sent_mask = [1] * min(len(sent), MAX_WORDS) + [0] * pad_length
        attention_masks.append(sent_mask)
        
        # Sentence mask
        if len(sent) > 0:
            sentence_masks.append(1)
        else:
            sentence_masks.append(0)
    
    # Final padding for sentences (already handled above)
    
    # Convert to tensors
    input_ids_tensor = torch.tensor([encoded_input_ids], dtype=torch.long)  # [1, num_sentences, seq_length]
    pos_tags_tensor = torch.tensor([encoded_pos_tags], dtype=torch.long)    # [1, num_sentences, seq_length]
    rules_tensor = torch.tensor([encoded_rules], dtype=torch.long)          # [1, num_sentences, seq_length, max_rules]
    attention_mask_tensor = torch.tensor([attention_masks], dtype=torch.float)  # [1, num_sentences, seq_length]
    sentence_masks_tensor = torch.tensor([sentence_masks], dtype=torch.float)  # [1, num_sentences]
    
    return input_ids_tensor, pos_tags_tensor, rules_tensor, attention_mask_tensor, sentence_masks_tensor

def main():
    # Load vocabularies and label mappings
    word_vocab, pos_vocab, rule_vocab, target_names = load_vocab()
    
    # Initialize the model
    vocab_size = len(word_vocab)
    pos_vocab_size = len(pos_vocab)
    rule_vocab_size = len(rule_vocab)
    num_classes = len(target_names)
    
    word_encoder_params = {
        'model_dim': 128,
        'num_heads': 4,
        'ff_dim': 512,
        'num_layers': 2,
        'dropout': 0.1,
    }
    sentence_encoder_params = {
        'model_dim': 128,  # Should match fusion_dim
        'num_heads': 4,
        'ff_dim': 512,
        'num_layers': 2,
        'dropout': 0.1,
    }
    
    model = get_model(
        vocab_size=vocab_size,
        pos_vocab_size=pos_vocab_size,
        rule_vocab_size=rule_vocab_size,
        num_classes=num_classes,
        embed_dim=100,
        pos_embed_dim=25,
        rule_embed_dim=25,
        fusion_dim=128,
        word_encoder_params=word_encoder_params,
        sentence_encoder_params=sentence_encoder_params,
        max_word_len=50,
        max_sent_len=30,
        max_rules_per_word=MAX_RULES_PER_WORD
    )
    
    model = model.to(DEVICE)
    
    # Load the trained model weights
    model_path = os.path.join(DATA_DIR, 'best_model.pt')
    if not os.path.exists(model_path):
        print(f"Model file {model_path} not found. Please train the model first.")
        return
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    print("Model loaded successfully.")
    
    # User input loop
    print("\nEnter text to classify (type 'exit' to quit):\n")
    while True:
        user_input = input(">> ")
        if user_input.lower() in ['exit', 'quit']:
            print("Exiting.")
            break
        if not user_input.strip():
            print("Empty input. Please enter valid text.")
            continue
        
        # Preprocess the input text
        input_ids, pos_tags, rules, attention_mask, sentence_masks = preprocess_text(
            user_input, word_vocab, pos_vocab, rule_vocab
        )
        
        # Move tensors to device
        input_ids = input_ids.to(DEVICE)
        pos_tags = pos_tags.to(DEVICE)
        rules = rules.to(DEVICE)
        attention_mask = attention_mask.to(DEVICE)
        sentence_masks = sentence_masks.to(DEVICE)
        
        # Perform prediction
        with torch.no_grad():
            outputs = model(input_ids, attention_mask, pos_tags, rules, sentence_masks)  # [1, num_classes]
            probs = torch.softmax(outputs, dim=1)
            _, pred = torch.max(probs, dim=1)
            pred = pred.item()
            confidence = probs[0][pred].item()
            predicted_label = target_names[pred]  # Correct mapping
            print(f"Predicted Class: {predicted_label} (Confidence: {confidence:.4f})\n")

if __name__ == "__main__":
    main()
