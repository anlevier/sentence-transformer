# Sentence Transformer Demo

This take home assignment uses Hugging Face's <code>distilbert-base-uncased</code> model and PyTorch to incode sentences into fixed-length embeddings using mean pooling.

---

## Setup

Install dependencies

```bash
pip install -r requirements.txt
```

(This downgrades NumPy to 2.x)

## Usage

```bash
python demo.py "Hello, World!" "Hello again, World!" [...]
```
