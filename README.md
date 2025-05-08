# Sentence Transformer Demo

This take home assignment uses Hugging Face's <code>distilbert-base-uncased</code> model and PyTorch to encode sentences into fixed-length embeddings using max pooling.

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

## Sample Output

```bash
python3 demo.py "Hello, World" "Hello again, World"

Sentence: Hello, World
Shape: torch.Size([768])
Embedding: tensor([ 9.1805e-01,  4.0750e-01,  1.0483e+00,  4.2093e-01,  4.3696e-01,
         1.8795e-01,  7.3252e-01,  1.3822e+00,  4.5166e-01,  6.0908e-02,
         1.3424e-01, -7.2506e-02,  1.0813e-01,  1.1419e+00,  1.5249e-01,
         ])


Sentence: Hello again, World
Shape: torch.Size([768])
Embedding: tensor([ 9.2065e-01,  2.3182e-01,  1.0902e+00,  3.8391e-01,  4.5512e-01,
         2.4818e-01,  9.6792e-01,  1.3265e+00,  4.0740e-01, -9.1656e-03,
         1.2341e-01, -6.9042e-02,  1.3175e-01,  1.1637e+00,  1.7635e-01,
         ])
```

Above embeddings are truncated