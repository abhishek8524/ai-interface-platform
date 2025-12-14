# Fake News Detection API

This project provides a FastAPI-based service for detecting fake news using a PyTorch BiLSTM model.

## API Endpoint

### POST /predict

**Request**
```json
{
  "text": "News article text"
}
