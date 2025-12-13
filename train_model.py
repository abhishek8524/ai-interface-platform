"""
Fake News Detection using LSTM (PyTorch)
This script trains a PyTorch LSTM model for detecting fake news from text data.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import re
import warnings
warnings.filterwarnings('ignore')

# PyTorch libraries
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import pickle

print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
print("\n" + "="*80)

# ============================================================================
# 1. LOAD AND EXPLORE DATASET
# ============================================================================
print("\n1. LOADING DATASET...")
df = pd.read_csv('news_dataset.csv')

print(f"Dataset Shape: {df.shape}")
print(f"\nLabel Distribution:\n{df['label'].value_counts()}")
print(f"\nMissing Values:\n{df.isnull().sum()}")

# ============================================================================
# 2. DATA PREPROCESSING
# ============================================================================
print("\n2. PREPROCESSING DATA...")

# Handle missing values
print(f"Rows before removing nulls: {len(df)}")
df = df.dropna()
print(f"Rows after removing nulls: {len(df)}")
df = df.reset_index(drop=True)

# Text cleaning function
def clean_text(text):
    """Clean and preprocess text data"""
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Apply cleaning
df['cleaned_text'] = df['text'].apply(clean_text)
print("Text cleaning completed!")

# Encode labels: FAKE=0, REAL=1
df['label_encoded'] = df['label'].map({'FAKE': 0, 'REAL': 1})

# ============================================================================
# 3. TOKENIZATION AND VOCABULARY
# ============================================================================
print("\n3. BUILDING VOCABULARY...")

class Vocabulary:
    def __init__(self, max_size=10000):
        self.word2idx = {'<PAD>': 0, '<UNK>': 1}
        self.idx2word = {0: '<PAD>', 1: '<UNK>'}
        self.max_size = max_size
        
    def build_vocab(self, texts):
        word_counts = Counter()
        for text in texts:
            word_counts.update(text.split())
        
        # Get most common words
        most_common = word_counts.most_common(self.max_size - 2)
        
        for idx, (word, count) in enumerate(most_common, start=2):
            self.word2idx[word] = idx
            self.idx2word[idx] = word
            
    def encode(self, text, max_len=300):
        tokens = text.split()
        encoded = [self.word2idx.get(word, 1) for word in tokens]
        # Padding or truncating
        if len(encoded) < max_len:
            encoded += [0] * (max_len - len(encoded))
        else:
            encoded = encoded[:max_len]
        return encoded
    
    def __len__(self):
        return len(self.word2idx)

# ============================================================================
# 4. PREPARE DATA
# ============================================================================
print("\n4. PREPARING DATA FOR LSTM...")

X = df['cleaned_text'].values
y = df['label_encoded'].values

# Split data: 80% train, 20% test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.20, 
    random_state=42, 
    stratify=y
)

print(f"Training samples: {len(X_train)} (80%)")
print(f"Testing samples: {len(X_test)} (20%)")
print(f"Train label distribution: {np.bincount(y_train)}")
print(f"Test label distribution: {np.bincount(y_test)}")

# Build vocabulary
vocab = Vocabulary(max_size=10000)
vocab.build_vocab(X_train)
print(f"Vocabulary size: {len(vocab)}")

# Encode sequences
MAX_LEN = 300
X_train_encoded = [vocab.encode(text, MAX_LEN) for text in X_train]
X_test_encoded = [vocab.encode(text, MAX_LEN) for text in X_test]

# Convert to tensors
X_train_tensor = torch.LongTensor(X_train_encoded)
X_test_tensor = torch.LongTensor(X_test_encoded)
y_train_tensor = torch.FloatTensor(y_train)
y_test_tensor = torch.FloatTensor(y_test)

print(f"Training tensor shape: {X_train_tensor.shape}")
print(f"Testing tensor shape: {X_test_tensor.shape}")

# ============================================================================
# 5. CREATE DATASET AND DATALOADER
# ============================================================================
class NewsDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]

train_dataset = NewsDataset(X_train_tensor, y_train_tensor)
test_dataset = NewsDataset(X_test_tensor, y_test_tensor)

BATCH_SIZE = 32
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f"\nNumber of training batches: {len(train_loader)}")
print(f"Number of test batches: {len(test_loader)}")

# ============================================================================
# 6. BUILD LSTM MODEL
# ============================================================================
print("\n5. BUILDING LSTM MODEL...")

class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, 
                 bidirectional, dropout):
        super(LSTMClassifier, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, 
                           hidden_dim, 
                           num_layers=n_layers, 
                           bidirectional=bidirectional,
                           dropout=dropout if n_layers > 1 else 0,
                           batch_first=True)
        
        self.fc1 = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, text):
        # text shape: [batch_size, seq_len]
        embedded = self.dropout(self.embedding(text))
        # embedded shape: [batch_size, seq_len, embedding_dim]
        
        output, (hidden, cell) = self.lstm(embedded)
        # For bidirectional, concatenate the final forward and backward hidden states
        if self.lstm.bidirectional:
            hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        else:
            hidden = hidden[-1,:,:]
        # hidden shape: [batch_size, hidden_dim * num_directions]
        
        x = self.dropout(self.relu(self.fc1(hidden)))
        x = self.dropout(self.relu(self.fc2(x)))
        x = self.sigmoid(self.fc3(x))
        
        return x.squeeze(1)

# Model parameters
VOCAB_SIZE = len(vocab)
EMBEDDING_DIM = 128
HIDDEN_DIM = 64
OUTPUT_DIM = 1
N_LAYERS = 2
BIDIRECTIONAL = True
DROPOUT = 0.3

model = LSTMClassifier(VOCAB_SIZE, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM, 
                       N_LAYERS, BIDIRECTIONAL, DROPOUT)
model = model.to(device)

print(f"\nModel Architecture:")
print(model)
print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters()):,}")

# ============================================================================
# 7. TRAINING SETUP
# ============================================================================
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

def calculate_accuracy(preds, labels):
    rounded_preds = torch.round(preds)
    correct = (rounded_preds == labels).float()
    return correct.sum() / len(correct)

# ============================================================================
# 8. TRAIN THE MODEL
# ============================================================================
print("\n6. TRAINING MODEL...")

EPOCHS = 20
best_valid_loss = float('inf')

history = {
    'train_loss': [], 'train_acc': [],
    'val_loss': [], 'val_acc': []
}

# Split training data for validation
train_size = int(0.8 * len(train_dataset))
val_size = len(train_dataset) - train_size
train_subset, val_subset = torch.utils.data.random_split(train_dataset, [train_size, val_size])

train_loader_split = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False)

for epoch in range(EPOCHS):
    # Training phase
    model.train()
    train_loss = 0
    train_acc = 0
    
    for texts, labels in train_loader_split:
        texts, labels = texts.to(device), labels.to(device)
        
        optimizer.zero_grad()
        predictions = model(texts)
        loss = criterion(predictions, labels)
        acc = calculate_accuracy(predictions, labels)
        
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        train_acc += acc.item()
    
    # Validation phase
    model.eval()
    val_loss = 0
    val_acc = 0
    
    with torch.no_grad():
        for texts, labels in val_loader:
            texts, labels = texts.to(device), labels.to(device)
            predictions = model(texts)
            loss = criterion(predictions, labels)
            acc = calculate_accuracy(predictions, labels)
            
            val_loss += loss.item()
            val_acc += acc.item()
    
    train_loss /= len(train_loader_split)
    train_acc /= len(train_loader_split)
    val_loss /= len(val_loader)
    val_acc /= len(val_loader)
    
    history['train_loss'].append(train_loss)
    history['train_acc'].append(train_acc)
    history['val_loss'].append(val_loss)
    history['val_acc'].append(val_acc)
    
    print(f'Epoch {epoch+1}/{EPOCHS}:')
    print(f'  Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}%')
    print(f'  Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}%')
    
    # Early stopping
    if val_loss < best_valid_loss:
        best_valid_loss = val_loss
        torch.save(model.state_dict(), 'best_model.pt')
        print(f'  ✓ Model saved!')

print("\nTraining completed!")

# Load best model
model.load_state_dict(torch.load('best_model.pt'))

# ============================================================================
# 9. EVALUATE MODEL
# ============================================================================
print("\n7. EVALUATING MODEL...")

model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for texts, labels in test_loader:
        texts, labels = texts.to(device), labels.to(device)
        predictions = model(texts)
        all_preds.extend(predictions.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

all_preds = np.array(all_preds)
all_labels = np.array(all_labels)
y_pred = (all_preds > 0.5).astype(int)

# Calculate metrics
test_accuracy = accuracy_score(all_labels, y_pred)
from sklearn.metrics import precision_score, recall_score, f1_score

test_precision = precision_score(all_labels, y_pred)
test_recall = recall_score(all_labels, y_pred)
test_f1 = f1_score(all_labels, y_pred)

print("\n" + "="*80)
print("TEST SET PERFORMANCE")
print("="*80)
print(f"Test Accuracy: {test_accuracy*100:.2f}%")
print(f"Test Precision: {test_precision:.4f}")
print(f"Test Recall: {test_recall:.4f}")
print(f"Test F1-Score: {test_f1:.4f}")
print("="*80)

# Classification report
print("\nDetailed Classification Report:")
print("="*80)
print(classification_report(all_labels, y_pred, target_names=['FAKE', 'REAL'], digits=4))
print("="*80)

# Confusion Matrix
cm = confusion_matrix(all_labels, y_pred)
tn, fp, fn, tp = cm.ravel()

print("\nConfusion Matrix Breakdown:")
print(f"True Negatives (FAKE correctly identified): {tn}")
print(f"False Positives (FAKE predicted as REAL): {fp}")
print(f"False Negatives (REAL predicted as FAKE): {fn}")
print(f"True Positives (REAL correctly identified): {tp}")

# ============================================================================
# 10. SAVE MODEL AND VISUALIZATIONS
# ============================================================================
print("\n8. SAVING MODEL AND VISUALIZATIONS...")

# Save model
torch.save(model.state_dict(), 'fake_news_lstm_pytorch.pt')
print("✓ Model saved as 'fake_news_lstm_pytorch.pt'")

# Save vocabulary
with open('vocabulary.pkl', 'wb') as f:
    pickle.dump(vocab, f)
print("✓ Vocabulary saved as 'vocabulary.pkl'")

# Save training history
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history['train_acc'], label='Train Accuracy', linewidth=2)
plt.plot(history['val_acc'], label='Validation Accuracy', linewidth=2)
plt.title('Model Accuracy', fontsize=14, fontweight='bold')
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.legend()
plt.grid(alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(history['train_loss'], label='Train Loss', linewidth=2)
plt.plot(history['val_loss'], label='Validation Loss', linewidth=2)
plt.title('Model Loss', fontsize=14, fontweight='bold')
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('training_history_pytorch.png', dpi=300, bbox_inches='tight')
print("✓ Training history plot saved as 'training_history_pytorch.png'")

# Save confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
            xticklabels=['FAKE', 'REAL'], yticklabels=['FAKE', 'REAL'],
            annot_kws={'size': 16, 'weight': 'bold'})
plt.title('Confusion Matrix', fontsize=16, fontweight='bold', pad=20)
plt.ylabel('True Label', fontsize=12, fontweight='bold')
plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('confusion_matrix_pytorch.png', dpi=300, bbox_inches='tight')
print("✓ Confusion matrix saved as 'confusion_matrix_pytorch.png'")

# ============================================================================
# 11. TEST WITH SAMPLE PREDICTIONS
# ============================================================================
print("\n9. TESTING WITH SAMPLE PREDICTIONS...")

def predict_news(text, model, vocab, max_len=300):
    """Predict whether a news article is FAKE or REAL"""
    model.eval()
    cleaned = clean_text(text)
    encoded = vocab.encode(cleaned, max_len)
    tensor = torch.LongTensor([encoded]).to(device)
    
    with torch.no_grad():
        prediction = model(tensor).item()
    
    label = 'REAL' if prediction > 0.5 else 'FAKE'
    confidence = prediction if prediction > 0.5 else 1 - prediction
    return label, confidence

print("\n" + "="*80)
print("SAMPLE PREDICTIONS FROM TEST SET")
print("="*80)

for i in range(5):
    idx = np.random.randint(0, len(X_test))
    true_label = 'REAL' if y_test[idx] == 1 else 'FAKE'
    pred_label, confidence = predict_news(X_test[idx], model, vocab, MAX_LEN)
    
    print(f"\nSample {i+1}:")
    print(f"Text: {X_test[idx][:150]}...")
    print(f"True Label: {true_label}")
    print(f"Predicted: {pred_label} (Confidence: {confidence*100:.2f}%)")
    print(f"Status: {'✓ CORRECT' if true_label == pred_label else '✗ INCORRECT'}")
    print("-"*80)

# ============================================================================
# 12. FINAL SUMMARY
# ============================================================================
print("\n" + "="*80)
print("MODEL SUMMARY")
print("="*80)
print(f"\nFramework: PyTorch {torch.__version__}")
print(f"Architecture: Bidirectional LSTM")
print(f"Vocabulary Size: {len(vocab)}")
print(f"Max Sequence Length: {MAX_LEN}")
print(f"Embedding Dimension: {EMBEDDING_DIM}")
print(f"Hidden Dimension: {HIDDEN_DIM}")
print(f"Number of Layers: {N_LAYERS}")
print(f"\nTraining Samples: {len(X_train)} (80%)")
print(f"Testing Samples: {len(X_test)} (20%)")
print(f"\nFinal Test Accuracy: {test_accuracy*100:.2f}%")
print(f"Final Test Precision: {test_precision:.4f}")
print(f"Final Test Recall: {test_recall:.4f}")
print(f"Final Test F1-Score: {test_f1:.4f}")
print("\n" + "="*80)
print("✓ MODEL TRAINING AND EVALUATION COMPLETE!")
print("="*80)
