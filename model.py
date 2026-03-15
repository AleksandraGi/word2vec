import numpy as np

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