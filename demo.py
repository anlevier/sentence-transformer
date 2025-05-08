import sys
import torch
from models.tasks import TasksModel

# Label Maps
emotion_labels = ["sadness", "joy", "love", "anger", "fear", "surprise"]
sentiment_labels = ["negative", "positive"]

def main(sentences):
    model = TasksModel()
    model.load_state_dict(torch.load("tasks_model.pt", map_location="cpu"))
    model.eval()

    with torch.no_grad():
        logits_emotion, logits_sentiment = model(sentences)
        pred_emotion = torch.argmax(logits_emotion, dim=1)
        pred_sentiment = torch.argmax(logits_sentiment, dim=1)

    for sentence, e_idx, s_idx in zip(sentences, pred_emotion, pred_sentiment):
        print(f"\nSentence: {sentence}")
        print(f"Predicted Emotion: {emotion_labels[e_idx]}")
        print(f"Predicted Sentiment: {sentiment_labels[s_idx]}")

if __name__ == "__main__":
    if len(sys.argv) <= 1:
        print("Give me at least one sentence to encode!")
        sys.exit(1)

    sentences = sys.argv[1:]
    main(sentences)
