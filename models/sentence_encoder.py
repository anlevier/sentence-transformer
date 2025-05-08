import torch
import torch.nn as nn
from transformers import AutoModel as BackboneLoader, AutoTokenizer as SentenceTokenizer

class SentenceTransformerModel(nn.Module):
    def __init__(self, model_name: str = "distilbert-base-uncased"):
        super().__init__()
        # DistilBERT - lightweight version of BERT
        self.encoder = BackboneLoader.from_pretrained(model_name)
        self.tokenizer = SentenceTokenizer.from_pretrained(model_name)

    def max_pool(self, model_output, attention_mask):
        token_embeddings = model_output.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        
        # Use tiny number so they don't skew max
        token_embeddings[input_mask_expanded == 0] = -1e9
    
        return torch.max(token_embeddings, dim=1).values

    
    def encode(self, sentences):
        self.eval()
        inputs = self.tokenizer(
            sentences,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )
        with torch.no_grad():
            model_output = self.encoder(**inputs)
        sentence_embeddings = self.max_pool(model_output, inputs["attention_mask"])
        return sentence_embeddings