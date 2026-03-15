import numpy as np

from preprocess import (
    tokenize,
    build_vocab,
    generate_pairs,
    build_negative_distribution,
    sample_negatives
)

from model import Word2Vec, nearest_neighbors

def train(text, embedding_dim=20, window_size=2, num_negatives=3, learning_rate=0.025, epochs=5):
    # Convert raw text into word tokens
    tokens = tokenize(text)

    # Build the vocabulary and encode the text as word ids
    word_to_id, id_to_word, encoded, counts = build_vocab(tokens)

    # Generate positive skip-gram training pairs
    pairs = generate_pairs(encoded, window_size)

    # Probability distribution for negative sampling
    probs = build_negative_distribution(word_to_id, counts)

    # Create the model
    model = Word2Vec(len(word_to_id), embedding_dim)

    # Train for number of epochs
    for epoch in range(epochs):
        np.random.shuffle(pairs)
        total_loss = 0.0

        # For each positive pair, sample negatives and update the model
        for center_id, pos_id in pairs:
            neg_ids = sample_negatives(probs, num_negatives, {center_id, pos_id})
            loss = model.train_step(center_id, pos_id, neg_ids, learning_rate)
            total_loss += loss

        avg_loss = total_loss / len(pairs)
        print(f"Epoch {epoch + 1}: loss = {avg_loss:.4f}")

    return model, word_to_id, id_to_word

def main():
    # Read the text file
    # with open("sample_text2.txt", "r", encoding="utf-8") as f:         # second training text
    with open("sample_text.txt", "r", encoding="utf-8") as f:
        text = f.read()

    # Train the model
    model, word_to_id, id_to_word = train(text)
    # Get learned input embeddings (W_in) 
    embeddings = model.get_embeddings()

    # Example words for nearest neighbours
    # for word in ["mathematics", "science", "king", "beauty", "human"]:     # test sample for sample_text2.txt
    for word in ["cat", "dog", "king", "queen", "human"]:
        if word in word_to_id:
            print(f"\nNearest neighbors for '{word}':")
            for neighbor, score in nearest_neighbors(word, word_to_id, id_to_word, embeddings):
                print(f"  {neighbor:12s} {score:.4f}")

if __name__ == "__main__":
    main()