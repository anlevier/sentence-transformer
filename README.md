# Sentence Transformer Demo

This take home assignment uses Hugging Face's <code>distilbert-base-uncased</code> model and PyTorch to encode sentences into fixed-length embeddings using max pooling.

It extends to a model that predicts **emotion** and **sentiment**.

---

## Setup

Install dependencies

```bash
pip install -r requirements.txt
```

(This downgrades NumPy to 2.x)

## Training

```bash
python -m train.train
```

## Usage

```bash
python3 demo.py "I love soup." "I hate sandwiches."
```

## Sample Output

```bash
Sentence: I love soup.
Predicted Emotion: joy
Predicted Sentiment: positive

Sentence: I hate sandwiches.
Predicted Emotion: anger
Predicted Sentiment: negative
```
