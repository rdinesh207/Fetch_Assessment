import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import List, Tuple, Dict
from multi_task_model import *

class SentenceDataset(Dataset):
    def __init__(self, sentences: List[str], labels_a: List[int], labels_b: List[int], tokenizer):
        self.sentences = sentences
        self.labels_a = labels_a
        self.labels_b = labels_b
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        tokenized = self.tokenizer(self.sentences[idx],padding='max_length',truncation=True,max_length=128,return_tensors='pt')
        return {
            'input_ids': tokenized['input_ids'].squeeze(),
            'attention_mask': tokenized['attention_mask'].squeeze(),
            'label_a': torch.tensor(self.labels_a[idx]),
            'label_b': torch.tensor(self.labels_b[idx])
        }

def train_one_shot():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize model and tokenizer
    model = MultiTaskSentenceTransformer(model_name="bert-base-uncased",num_classes_task_a=3,num_classes_task_b=2,).to(device)
    
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    # One-shot learning examples (one per class)
    train_sentences = [
        "This product exceeded all my expectations!",  # Positive (0)
        "This is the worst purchase I've ever made.",  # Negative (1)
        "The product arrived on schedule.",           # Neutral (2)
    ]
    
    train_labels_a = [0, 1, 2]  # Sentiment labels
    train_labels_b = [0, 0, 1]  # Formality labels (0: informal, 1: formal)

    # Create training dataset
    train_dataset = SentenceDataset(
        sentences=train_sentences,
        labels_a=train_labels_a,
        labels_b=train_labels_b,
        tokenizer=tokenizer
    )

    # Create training dataloader
    train_loader = DataLoader(train_dataset, batch_size=3, shuffle=True)

    # Freeze the transformer backbone
    for param in model.backbone.parameters():
        param.requires_grad = False

    # Initialize optimizer with layer-wise learning rates
    optimizer = torch.optim.AdamW([
        {'params': model.pooling.parameters(), 'lr': 2e-5},
        {'params': model.task_a_head.parameters(), 'lr': 5e-5},
        {'params': model.task_b_head.parameters(), 'lr': 5e-5}
    ])

    # Training loop
    model.train()
    num_epochs = 50  # More epochs for one-shot learning
    
    print("\nStarting one-shot training...")
    
    for epoch in range(num_epochs):
        epoch_loss = 0
        for batch in train_loader:
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # Forward pass
            outputs = model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask']
            )
            
            # Calculate losses
            loss_a = F.cross_entropy(outputs['task_a_logits'], batch['label_a'])
            loss_b = F.cross_entropy(outputs['task_b_logits'], batch['label_b'])
            total_loss = loss_a + loss_b
            
            # Backward pass
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            epoch_loss += total_loss.item()
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}")

    print("Training completed!")
    return model, tokenizer

def test_model(model, tokenizer, device):
    # Test sentences
    test_sentences = [
        "This is absolutely fantastic!",                    # Similar to positive
        "I regret buying this completely.",                # Similar to negative
        "The package contains three items.",               # Similar to neutral
        "I've never been more disappointed.",              # Negative
        "The technical specifications are as follows:",    # Formal/Neutral
    ]
    
    # Tokenize test sentences
    inputs = tokenizer(
        test_sentences,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt"
    ).to(device)

    # Get predictions
    model.eval()
    with torch.no_grad():
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"]
        )

    # Process results
    task_a_preds = torch.argmax(outputs['task_a_logits'], dim=1).cpu().numpy()
    task_b_preds = torch.argmax(outputs['task_b_logits'], dim=1).cpu().numpy()
    
    # Labels for interpretation
    sentiment_labels = ['Positive', 'Negative', 'Neutral']
    formality_labels = ['Informal', 'Formal']

    # Print results
    print("\n=== Test Results ===")
    for i, sentence in enumerate(test_sentences):
        print(f"\nTest sentence: {sentence}")
        print(f"Predicted sentiment: {sentiment_labels[task_a_preds[i]]}")
        print(f"Predicted formality: {formality_labels[task_b_preds[i]]}")
        
        # Get probabilities
        sentiment_probs = F.softmax(outputs['task_a_logits'][i], dim=0).cpu().numpy()
        print("\nSentiment probabilities:")
        for label, prob in zip(sentiment_labels, sentiment_probs):
            print(f"{label}: {prob:.3f}")

def main():
    # Train the model
    model, tokenizer = train_one_shot()
    
    # Test the model
    device = next(model.parameters()).device
    test_model(model, tokenizer, device)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"An error occurred: {str(e)}")