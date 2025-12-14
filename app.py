from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
import torch.nn as nn
import pickle
import re
from collections import Counter
import vocab




# =========================
# FastAPI App
# =========================
app = FastAPI(title="Fake News Detection API")



# =========================
# CORS (AFTER app creation)
# =========================
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:4200",
        "https://*.onrender.com",
        "*"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)





# =========================
# Device
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# Text Cleaning (same as training)
# =========================
def clean_text(text: str):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# =========================
# Vocabulary Class (IMPORTANT)
# =========================
class Vocabulary:
    def __init__(self, max_size=10000):
        self.word2idx = {'<PAD>': 0, '<UNK>': 1}
        self.idx2word = {0: '<PAD>', 1: '<UNK>'}
        self.max_size = max_size

    def build_vocab(self, texts):
        word_counts = Counter()
        for text in texts:
            word_counts.update(text.split())

        most_common = word_counts.most_common(self.max_size - 2)
        for idx, (word, _) in enumerate(most_common, start=2):
            self.word2idx[word] = idx
            self.idx2word[idx] = word

    def encode(self, text, max_len=300):
        tokens = text.split()
        encoded = [self.word2idx.get(word, 1) for word in tokens]

        if len(encoded) < max_len:
            encoded += [0] * (max_len - len(encoded))
        else:
            encoded = encoded[:max_len]

        return encoded

    def __len__(self):
        return len(self.word2idx)

# =========================
# LSTM Model (same as training)
# =========================
class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim,
                 output_dim, n_layers, bidirectional, dropout):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=n_layers,
            bidirectional=bidirectional,
            dropout=dropout if n_layers > 1 else 0,
            batch_first=True
        )

        self.fc1 = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.sigmoid = nn.Sigmoid()

    def forward(self, text):
        embedded = self.dropout(self.embedding(text))
        _, (hidden, _) = self.lstm(embedded)

        if self.lstm.bidirectional:
            hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
        else:
            hidden = hidden[-1]

        x = self.dropout(self.relu(self.fc1(hidden)))
        x = self.dropout(self.relu(self.fc2(x)))
        x = self.sigmoid(self.fc3(x))

        return x.squeeze(1)

# =========================
# Load Vocabulary
# =========================
with open("vocabulary.pkl", "rb") as f:
    vocab = pickle.load(f)

MAX_LEN = 300

# =========================
# Load Trained Model
# =========================
model = LSTMClassifier(
    vocab_size=len(vocab),
    embedding_dim=128,
    hidden_dim=64,
    output_dim=1,
    n_layers=2,
    bidirectional=True,
    dropout=0.3
)

model.load_state_dict(
    torch.load("fake_news_lstm_pytorch.pt", map_location=device)
)
model.to(device)
model.eval()

# =========================
# Request Schema
# =========================
class NewsInput(BaseModel):
    text: str

# =========================
# Routes
# =========================
@app.get("/")
def home():
    return {"message": "Fake News Detection API is running 🚀"}


@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict_news(news: NewsInput):
    cleaned = clean_text(news.text)

    # -------------------------
    # 1️⃣ Empty or very short text check
    # -------------------------
    if not cleaned or len(cleaned.split()) < 5:
        return {
            "prediction": "UNCERTAIN",
            "confidence": 50.0,
            "note": "insufficient or meaningless text"
        }

    # -------------------------
    # 2️⃣ RULE-BASED SAFETY NET
    # -------------------------
    fake_keywords = [
        "aliens", "time travel", "lizard people",
        "mind control", "secret government",
        "instant cure", "miracle", "conspiracy"
    ]

    if any(k in cleaned for k in fake_keywords):
        return {
            "prediction": "FAKE",
            "confidence": 99.0,
            "note": "rule-based override"
        }

    # -------------------------
    # 3️⃣ Model inference
    # -------------------------
    encoded = vocab.encode(cleaned, MAX_LEN)
    tensor = torch.LongTensor([encoded]).to(device)

    with torch.no_grad():
        prediction = model(tensor).item()

    # -------------------------
    # 4️⃣ Confidence gating
    # -------------------------
    if prediction > 0.9:
        label = "REAL"
        confidence = prediction
    elif prediction < 0.1:
        label = "FAKE"
        confidence = 1 - prediction
    else:
        label = "UNCERTAIN"
        confidence = 0.5

    return {
        "prediction": label,
        "confidence": round(confidence * 100, 2)
    }


   
