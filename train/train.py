import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import load_dataset
from torch.optim import AdamW
from sklearn.metrics import accuracy_score
from models.tasks import TasksModel

# Settings
BATCH_SIZE = 32
EPOCHS = 3
LR = 2e-5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Emotion | Sentiment
emotion_to_sentiment = {
    0: 0,  # sadness | negative
    1: 1,  # joy | positive
    2: 1,  # love | positive
    3: 0,  # anger | negative
    4: 0,  # fear | negative
    5: 0   # surprise | negative
}

# Load dataset
data = load_dataset("emotion", split="train[:1000]")
sentences = data["text"]
emotion_labels = torch.tensor(data["label"])
sentiment_labels = torch.tensor([emotion_to_sentiment[e] for e in data["label"]])

# Prepare DataLoader
dataset = list(zip(sentences, emotion_labels, sentiment_labels))
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# Initialize model
model = TasksModel(freeze=False).to(DEVICE)
loss_fn = nn.CrossEntropyLoss()
optimizer = AdamW(model.parameters(), lr=LR)

# Training loop
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    all_preds_e, all_labels_e = [], []
    all_preds_s, all_labels_s = [], []

    for batch in dataloader:
        batch_sentences, batch_emotions, batch_sentiments = batch
        batch_emotions = batch_emotions.to(DEVICE)
        batch_sentiments = batch_sentiments.to(DEVICE)

        optimizer.zero_grad()
        logits_emotion, logits_sentiment = model(batch_sentences)

        loss_emotion = loss_fn(logits_emotion, batch_emotions)
        loss_sentiment = loss_fn(logits_sentiment, batch_sentiments)
        loss = loss_emotion + loss_sentiment

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        all_preds_e.extend(logits_emotion.argmax(dim=1).cpu())
        all_labels_e.extend(batch_emotions.cpu())
        all_preds_s.extend(logits_sentiment.argmax(dim=1).cpu())
        all_labels_s.extend(batch_sentiments.cpu())

    acc_emotion = accuracy_score(all_labels_e, all_preds_e)
    acc_sentiment = accuracy_score(all_labels_s, all_preds_s)

    print(f"\nEpoch {epoch + 1}")
    print(f"Total Loss:         {total_loss:.4f}")
    print(f"Emotion Accuracy:   {acc_emotion:.4f}")
    print(f"Sentiment Accuracy: {acc_sentiment:.4f}")

# Save trained model
torch.save(model.state_dict(), "tasks_model.pt")
print("\n Trained model saved to tasks_model.pt")
