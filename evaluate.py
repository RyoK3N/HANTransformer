# evaluation.py

import os
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from model import get_model  # Ensure model.py is in the same directory
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

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
BATCH_SIZE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_RULES_PER_WORD = 3  # Should match preprocessing

class NewsgroupsDataset(Dataset):
    def __init__(self, data_split):
        """
        Initializes the dataset with the given data split ('train' or 'test').
        """
        self.input_ids = torch.tensor(data_split['input_ids'], dtype=torch.long)
        self.pos_tags = torch.tensor(data_split['pos_tags'], dtype=torch.long)
        self.rules = torch.tensor(data_split['rules'], dtype=torch.long)
        self.attention_mask = torch.tensor(data_split['attention_mask'], dtype=torch.float)
        self.sentence_masks = torch.tensor(data_split['sentence_masks'], dtype=torch.float)
        self.labels = torch.tensor(data_split['labels'], dtype=torch.long)
    
    def __len__(self):
        return self.input_ids.size(0)
    
    def __getitem__(self, idx):
        return {
            'input_ids': self.input_ids[idx],               # [num_sentences, seq_length]
            'pos_tags': self.pos_tags[idx],                 # [num_sentences, seq_length]
            'rules': self.rules[idx],                       # [num_sentences, seq_length, max_rules]
            'attention_mask': self.attention_mask[idx],     # [num_sentences, seq_length]
            'sentence_masks': self.sentence_masks[idx],     # [num_sentences]
            'labels': self.labels[idx]                      # scalar
        }

def collate_fn(batch):
    """
    Collate function to combine samples into a batch.
    """
    return {
        'input_ids': torch.stack([item['input_ids'] for item in batch]),
        'pos_tags': torch.stack([item['pos_tags'] for item in batch]),
        'rules': torch.stack([item['rules'] for item in batch]),
        'attention_mask': torch.stack([item['attention_mask'] for item in batch]),
        'sentence_masks': torch.stack([item['sentence_masks'] for item in batch]),
        'labels': torch.stack([item['labels'] for item in batch])
    }

def load_data():
    """
    Loads the processed data from the JSON file.
    """
    with open(PROCESSED_DATA_PATH, 'r') as f:
        data = json.load(f)
    return data

def evaluate(model, dataloader, criterion, device):
    """
    Evaluates the model on the given dataloader.
    Returns average loss and accuracy.
    """
    model.eval()
    epoch_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)                 # [batch_size, num_sentences, seq_length]
            pos_tags = batch['pos_tags'].to(device)                   # [batch_size, num_sentences, seq_length]
            rules = batch['rules'].to(device)                         # [batch_size, num_sentences, seq_length, max_rules]
            attention_mask = batch['attention_mask'].to(device)       # [batch_size, num_sentences, seq_length]
            sentence_masks = batch['sentence_masks'].to(device)       # [batch_size, num_sentences]
            labels = batch['labels'].to(device)                       # [batch_size]
            
            outputs = model(input_ids, attention_mask, pos_tags, rules, sentence_masks)  # [batch_size, num_classes]
            loss = criterion(outputs, labels)
            epoch_loss += loss.item()
            
            preds = torch.argmax(outputs, dim=1).detach().cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.detach().cpu().numpy())
    
    avg_loss = epoch_loss / len(dataloader)
    acc = accuracy_score(all_labels, all_preds)
    return avg_loss, acc, all_labels, all_preds

def main():
    # Load data
    print("Loading preprocessed data...")
    data = load_data()
    test_data = data['test']
    vocab = data['vocab']
    num_classes = len(vocab['label_to_id'])
    
    # Create dataset and dataloader
    print("Creating dataset and dataloader...")
    test_dataset = NewsgroupsDataset(test_data)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    
    # Initialize the model
    print("Initializing the model...")
    vocab_size = len(vocab['word_vocab'])
    pos_vocab_size = len(vocab['pos_vocab'])
    rule_vocab_size = len(vocab['rule_vocab'])
    
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
    
    # Define loss criterion
    criterion = nn.CrossEntropyLoss()
    
    # Evaluate the model
    print("Evaluating the model on the test set...")
    test_loss, test_acc, all_labels, all_preds = evaluate(model, test_loader, criterion, DEVICE)
    print(f"\nTest Loss: {test_loss:.4f} | Test Accuracy: {test_acc:.4f}")
    
    # Detailed classification report
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=vocab['target_names']))
    
    # Confusion Matrix
    print("Confusion Matrix:")
    cm = confusion_matrix(all_labels, all_preds)
    print(cm)

if __name__ == "__main__":
    main()
