import torch.nn as nn
from models.sentence_encoder import SentenceTransformerModel

class TasksModel(nn.Module):
    def __init__(self, freeze: bool = False):
        super().__init__()
        self.encoder = SentenceTransformerModel()

        if freeze:
            for param in self.encoder.parameters():
                param.requires_grad = False

        self.emotion_head = nn.Linear(768, 6)
        self.sentiment_head = nn.Linear(768, 2)

    def forward(self, sentences):
        embeddings = self.encoder(sentences)
        logits_emotion = self.emotion_head(embeddings)
        logits_sentiment = self.sentiment_head(embeddings)
        return logits_emotion, logits_sentiment