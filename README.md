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

# Code Review

## Task 1 - Sentence Transformer Implementation

"Test your implementation with a few sample sentences and showcase the obtained embeddings."

I tackled this task piece by piece using an iterative approach, so my code changed over commits. You can see the raw vectors in this previous version of the [README](https://github.com/anlevier/sentence-transformer/blob/426150d1f558471fc3aec4d7f373b3337e978aa0/README.md). In an intermediate step, I thought better of dumping all the vectors and [truncated the results](https://github.com/anlevier/sentence-transformer/blob/f76dda2f1b7c74bf03d81315a0e1aecadd04692a/README.md).

I used max pooling to create fixed-length embeddings for each sentence. Because this is a small demo and I made the choice to have users input the sentences in the command line, I assumed sentences would be simple and concise. Max pooling selects the max value, or strongest feature, across each embedding for the tokens in a sentence.

## Task 2 - Multi-Task Learning Expansion

- Task A - Sentence Classification - Emotion

I first classify each sentence into an emotion category:

```bash
[sadness, joy, love, anger, fear, surprise]
```
- Task B - Sentence Classification - Sentiment

I then classify each sentence into positive or negative sentiment

The changes to accomodate this appear in [this commit](https://github.com/anlevier/sentence-transformer/commit/a00d076e7cb494add343aa1525784a337f8a7b4e).

With the above changes there are now two linear layers:

```bash
self.emotion_head = nn.Linear(768, 6) # emotion logits.

self.sentiment_head = nn.Linear(768, 2) # sentiment logits.
```
Both layers share the encoder, reducing trainable parameters while allowing each task to retain flexibility about their own decision boundaries.

## Task 3 - Training Considerations

1. If the entire network should be frozen.

It could be useful to freeze the entire network if the model has already been fully trained for a specific task. This allows you to use less resources to make inferences and has predictable outputs.

2. If only the transformer backbone should be frozen.

When you're beginning to fine-tune it may be effective to freeze just the transformer backbone. You can focus on training specific tasks like `sentiment_head` or `emotion_head` without running the risk of overfitting.

3. If only one of the task-specific heads (either for Task A or Task B) should be frozen.

Once you have fine-tuned one task, it would make sense to freeze it to stop it from changing while you fine-tune the other. This is also useful when creating a new task.

## Task 4 - Training Loop Implementation (BONUS)

I implemented the training loop to optimize both classification tasks (emotion and sentiment). I used the HF `emotion` dataset, but since it doesn't provide sentiment labels I mapped a binary sentiment to each emotion.

[# Emotion | Sentiment](https://github.com/anlevier/sentence-transformer/blob/a00d076e7cb494add343aa1525784a337f8a7b4e/train/train.py#L16)
```bash
emotion_to_sentiment = {
    0: 0,  # sadness | negative
    1: 1,  # joy | positive
    2: 1,  # love | positive
    3: 0,  # anger | negative
    4: 0,  # fear | negative
    5: 0   # surprise | negative
}
```
The sentences are passed through the shared `SentenceTransformerModel` which creates fixed-length embeddings. The embeddings are then passed into the task-specific heads, `emotion_head` and `sentiment_head`.

Each head has its own cross-entropy loss function that are then combined into a scalar.

After each training epoch I compute classification accuracy for both heads using `sklearn.metrics.accuracy_score`.

I know, I didn't **have** to train, but I've done very similar work in a recent project so it was a relatively easy addition for a little ✨razzmatazz✨.
