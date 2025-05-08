import sys
from models.sentence_encoder import SentenceTransformerModel

def main(sentences):
    model = SentenceTransformerModel()

    embeddings = model.encode(sentences)

    for sentence, embedding in zip(sentences, embeddings):
        print(f"Shape: {embedding.shape}")
        print(f"Embedding: {embedding}\n")

if __name__ == "__main__":
    if len(sys.argv) <= 1:
        print("Give me at least one sentence to encode!")
        sys.exit(1)

    sentences = sys.argv[1:]

    main(sentences)    
