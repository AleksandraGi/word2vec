import re
from collections import Counter
import numpy as np

def tokenize(text):
    # Convert text to lowercase and extract words made of letters a-z
    # Output: a list of word tokens
    return re.findall(r"\b[a-z]+\b", text.lower())

def build_vocab(tokens, min_count=1):
    # Count how many times each word occurs
    counts = Counter(tokens)

    # Keep only words that appear at least min_count times
    vocabulary = [word for word, count in counts.items() if count >= min_count]
    vocabulary.sort()

    # Create mappings between words and integer ids
    # word -> index
    word_to_id = {word: i for i, word in enumerate(vocabulary)}
    # index -> word
    id_to_word = {i: word for word, i in word_to_id.items()}

    # Encode text into list of word ids
    encoded = [word_to_id[word] for word in tokens if word in word_to_id]

    return word_to_id, id_to_word, encoded, counts

def generate_pairs(token_ids, window_size=2):
    pairs = []

    for i, center_id in enumerate(token_ids):
        # Define the boundaries of the context window
        left = max(0, i - window_size)
        right = min(len(token_ids), i + window_size + 1)

        # Add (center, context) pairs for all words in the window
        for j in range(left, right):
            if j != i:
                pairs.append((center_id, token_ids[j]))

    return pairs

def build_negative_distribution(word_to_id, counts):
    # Create an array of word frequencies aligned with word ids (array contains zeros)
    frequencies = np.zeros(len(word_to_id), dtype=np.float64)

    # Fill the array with word counts
    for word, idx in word_to_id.items():
        frequencies[idx] = counts[word]

    # Use the unigram**0.75 distribution:
    # common words are sampled more often, but extremely frequent words become less dominant than with raw counts
    probs = frequencies ** 0.75
    # Normalize the values so they sum to 1 
    probs /= probs.sum()
    return probs

def sample_negatives(probs, num_negatives, forbidden):
    # Randomly sample words to use as negative context examples
    negatives = []
    while len(negatives) < num_negatives:
        # Sample one word id from the negative sampling distribution
        sample = np.random.choice(len(probs), p=probs)
        # Skip the center word and the positive context word
        if sample not in forbidden:
            negatives.append(sample)

    return negatives

class Word2Vec:
    def __init__(self, vocab_size, embedding_dim, seed=42):
        # Random number generator
        rng = np.random.default_rng(seed)

        # Two embedding matrices:
        # W_in  -> embeddings for center words
        # W_out -> embeddings for context and negative words
        self.W_in = 0.01 * rng.standard_normal((vocab_size, embedding_dim))
        self.W_out = 0.01 * rng.standard_normal((vocab_size, embedding_dim))

    def sigmoid(self, x):
        # Clip values for numerical stability
        x = np.clip(x, -10, 10)
        # Sigmoid equation
        return 1.0 / (1.0 + np.exp(-x))

    def train_step(self, center_id, pos_id, neg_ids, learning_rate):
        # Embedding of a central word 
        v = self.W_in[center_id].copy()
        # Embedding of a positive context word 
        u_pos = self.W_out[pos_id].copy()
        # Embeddings of a negative words
        u_neg = self.W_out[neg_ids].copy()

        # Scores for the positive pair and negative pairs
        pos_score = np.dot(u_pos, v)
        neg_scores = u_neg @ v

        # Apply sigmoid to convert scores into values vetween 0 and 1
        pos_sig = self.sigmoid(pos_score)
        neg_sig = self.sigmoid(neg_scores)

        # SGNS loss and one training example
        eps = 1e-10
        loss = -np.log(pos_sig + eps) - np.sum(np.log(self.sigmoid(-neg_scores) + eps))

        # Gradients of the loss
        grad_pos = pos_sig - 1.0
        grad_neg = neg_sig

        # Gradients with respect to the embeddings
        grad_v = grad_pos * u_pos + np.sum(grad_neg[:, None] * u_neg, axis=0)
        grad_u_pos = grad_pos * v
        grad_u_neg = grad_neg[:, None] * v[None, :]

        # Update the embeddings
        self.W_in[center_id] -= learning_rate * grad_v
        self.W_out[pos_id] -= learning_rate * grad_u_pos
        self.W_out[neg_ids] -= learning_rate * grad_u_neg

        return float(loss)

    def get_embeddings(self):
        # Return the learned input embeddings
        return self.W_in

def normalize_rows(matrix):
    # Normalize each embedding vector to length 1
    norms = np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-10
    return matrix / norms

def nearest_neighbors(query_word, word_to_id, id_to_word, embeddings, top_k=5):
    # Find the most similar words to the query word

    # If the word is not in the vocabulary, return an empty list
    if query_word not in word_to_id:
        return []

    normalized = normalize_rows(embeddings)

    # Get the id and embedding of the query word
    query_id = word_to_id[query_word]
    query_vector = normalized[query_id]

    # Compute similarity between the query word and all vocabulary words
    similarities = normalized @ query_vector
    # Sort ids by similarity in descending order
    best_ids = np.argsort(-similarities)

    results = []
    for idx in best_ids:
        # Skip the query word
        if idx == query_id:
            continue

        # Convert the id back to a word and store its similarity score
        results.append((id_to_word[idx], float(similarities[idx])))
        if len(results) >= top_k:
            break

    return results

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