import re
from collections import Counter
import numpy as np

def tokenize(text):
    # change text into lowercase, find words composed of the letters a-z, output: list of words
    return re.findall(r"\b[a-z]+\b", text.lower())

def build_vocab(tokens, min_count=1):
    # how many times each word occured
    counts = Counter(tokens)

    # leaves words that occure enough times (important if min_count != 1)
    vocabulary = [word for word, count in counts.items() if count >= min_count]
    vocabulary.sort()

    # mapping words
    # word -> index
    word_to_id = {word: i for i, word in enumerate(vocabulary)}
    # index -> word
    id_to_word = {i: word for word, i in word_to_id.items()}

    # change text into list of numbers
    encoded = [word_to_id[word] for word in tokens if word in word_to_id]

    return word_to_id, id_to_word, encoded, counts

def generate_pairs(token_ids, window_size=2):
    pairs = []

    for i, center_id in enumerate(token_ids):
        # boundries of context window
        left = max(0, i - window_size)
        right = min(len(token_ids), i + window_size + 1)

        # adding to pairs[] pairs of central word and its context words
        for j in range(left, right):
            if j != i:
                pairs.append((center_id, token_ids[j]))

    return pairs

def build_negative_distribution(word_to_id, counts):
    # list of zeros (size as the size of dictionary)
    frequencies = np.zeros(len(word_to_id), dtype=np.float64)

    # iterates over each word in dictionary and checks its frequency
    for word, idx in word_to_id.items():
        frequencies[idx] = counts[word]

    # reduce the dominance of extremely frequent words while still sampling common words more often than rare ones 
    probs = frequencies ** 0.75
    # normalization 
    probs /= probs.sum()
    return probs

def sample_negatives(probs, num_negatives, forbidden):
    # randomly pick a few words from the vocabulary to act as false context examples
    # list of negatives' indexes
    negatives = []
    while len(negatives) < num_negatives:
        # gets random index
        sample = np.random.choice(len(probs), p=probs)
        # if word isn't centralk or positive word
        if sample not in forbidden:
            negatives.append(sample)

    return negatives

class Word2Vec:
    def __init__(self, vocab_size, embedding_dim, seed=42):
        # random number generator
        rng = np.random.default_rng(seed)
        # matrix of central words and matrix of context and negative words
        self.W_in = 0.01 * rng.standard_normal((vocab_size, embedding_dim))
        self.W_out = 0.01 * rng.standard_normal((vocab_size, embedding_dim))

    def sigmoid(self, x):
        # transform score to range (0 - 1)
        x = np.clip(x, -10, 10)
        # sigmoid equation
        return 1.0 / (1.0 + np.exp(-x))

    def train_step(self, center_id, pos_id, neg_ids, learning_rate):
        # embedding of a central word 
        v = self.W_in[center_id].copy()
        # embedding of a context word 
        u_pos = self.W_out[pos_id].copy()
        # embeddings of a negative words
        u_neg = self.W_out[neg_ids].copy()

        # scores for positive ang negative words
        pos_score = np.dot(u_pos, v)
        neg_scores = u_neg @ v

        pos_sig = self.sigmoid(pos_score)
        neg_sig = self.sigmoid(neg_scores)

        # SGNS loss for single sample
        eps = 1e-10
        loss = -np.log(pos_sig + eps) - np.sum(np.log(self.sigmoid(-neg_scores) + eps))

        # gradients
        # derivative of the loss after pos_score
        grad_pos = pos_sig - 1.0
        # derivative of the loss after neg_scores
        grad_neg = neg_sig

        # gradient to central embedding
        grad_v = grad_pos * u_pos + np.sum(grad_neg[:, None] * u_neg, axis=0)
        # gradient to positive context embedding
        grad_u_pos = grad_pos * v
        # gradient to negatives embeddings
        grad_u_neg = grad_neg[:, None] * v[None, :]

        # embeddings update
        self.W_in[center_id] -= learning_rate * grad_v
        self.W_out[pos_id] -= learning_rate * grad_u_pos
        self.W_out[neg_ids] -= learning_rate * grad_u_neg

        return float(loss)

    def get_embeddings(self):
        return self.W_in

def normalize_rows(matrix):
    norms = np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-10
    return matrix / norms

def nearest_neighbors(query_word, word_to_id, id_to_word, embeddings, top_similarities=5):
    # finds words the most similar to query word

    # if query word is not in dictionary we can't find similar words 
    if query_word not in word_to_id:
        return []

    normalized = normalize_rows(embeddings)
    # index of query word
    query_id = word_to_id[query_word]
    # embedding of query word
    query_vector = normalized[query_id]

    # similarity value for each word in the vocabulary 
    similarities = normalized @ query_vector
    # sorted similarities
    best_ids = np.argsort(-similarities)

    results = []
    for idx in best_ids:
        # ignore query word
        if idx == query_id:
            continue

        # switch id to word, take its simlarity and add it do results[]
        results.append((id_to_word[idx], float(similarities[idx])))
        if len(results) >= top_similarities:
            break

    return results

def train(text, embedding_dim=20, window_size=2, num_negatives=3, learning_rate=0.025, epochs=5):
    # transform text inot list of words
    tokens = tokenize(text)
    # build a dictionary and code textr
    word_to_id, id_to_word, encoded, counts = build_vocab(tokens)

    # generate training pairs for skip-gram 
    pairs = generate_pairs(encoded, window_size)
    # probability distribution
    probs = build_negative_distribution(word_to_id, counts)

    # create a model
    model = Word2Vec(len(word_to_id), embedding_dim)

    # iterate through epochs
    for epoch in range(epochs):
        total_loss = 0.0

        # for each positive pair draw negatives, count loss and update it
        for center_id, pos_id in pairs:
            neg_ids = sample_negatives(probs, num_negatives, {center_id, pos_id})
            loss = model.train_step(center_id, pos_id, neg_ids, learning_rate)
            total_loss += loss

        avg_loss = total_loss / len(pairs)
        print(f"Epoch {epoch + 1}: loss = {avg_loss:.4f}")

    return model, word_to_id, id_to_word

def main():
    # read text file
    with open("sample_corpus.txt", "r", encoding="utf-8") as f:
        text = f.read()

    # start training
    model, word_to_id, id_to_word = train(text)
    # gets embeddings (W_in) 
    embeddings = model.get_embeddings()

    # random words to find its nearest neighbours
    for word in ["cat", "dog", "king", "queen", "human"]:
        if word in word_to_id:
            print(f"\nNearest neighbors for '{word}':")
            for neighbor, score in nearest_neighbors(word, word_to_id, id_to_word, embeddings):
                print(f"  {neighbor:12s} {score:.4f}")

if __name__ == "__main__":
    main()