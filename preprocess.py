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