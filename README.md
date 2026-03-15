# word2vec

This is a minimal implementation of the core word2vec training loop in pure NumPy, using skip-gram with negative sampling. The focus is on implementing the forward pass, loss, gradients, and parameter updates from scratch rather than building a production-ready training pipeline.

The skip-gram variant with negative sampling were implemenmted. The goal was to understand and reproduce the forward pass, loss, gradients, and parameter updates from scratch, without using PyTorch or TensorFlow.

## What is included
- basic text preprocessing
- vocabulary construction
- skip-gram pair generation
- negative sampling
- manual loss and gradients

## def tokenize(text)
Change text into lowercase, find words composed of the letters a-z.

## def build_vocab(tokens, min_count=1) 
This function creates the vocabulary and turns the text into numbers so the model can work with it.

## def generate_pairs(token_ids, window_size=2)
generate_pairs() creates skip-gram training examples by pairing each center word with all context words inside a fixed window around it.

## def build_negative_distribution(word_to_id, counts)
build_negative_distribution() creates the probability distribution used for negative sampling from word frequencies, using the standard unigram^0.75 weighting.

## def sample_negatives(probs, num_negatives, forbidden)
This function draws a small set of negative word ids from the sampling distribution while avoiding forbidden ids such as the center word and the positive context word.

## class Worde2Vec
### def __init__(self, vocab_size, embedding_dim, seed=42)
The constructor initializes two embedding matrices: one for center words (W_in) and one for context words (W_out). Both are randomly initialized with small values.

### def train_step(self, center_id, pos_id, neg_ids, learning_rate)
In train_step, I take one center word, one positive context word, and several negative samples. I compute the positive and negative dot products, apply the sigmoid function, compute the SGNS loss, derive the gradients manually, and update only the embeddings that participate in the current sample.

## normalize_rows, nearest_neighbours
After training, the input embeddings are used to inspect nearest neighbors based on cosine similarity, which gives a simple qualitative check of whether the learned representations make sense.

## def train()
train() prepares the data, initializes the model, iterates over the training pairs for several epochs, samples negatives for each pair, and repeatedly calls train_step() to update the embeddings.

