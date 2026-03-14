# word2Vec

This is a minimal implementation of the core word2vec training loop in pure NumPy, using skip-gram with negative sampling. The focus is on implementing the forward pass, loss, gradients, and parameter updates from scratch rather than building a production-ready training pipeline.

The skip-gram variant with negative sampling were implemenmted. The goal was to understand and reproduce the forward pass, loss, gradients, and parameter updates from scratch, without using PyTorch or TensorFlow.

## What is included
- basic text preprocessing
- vocabulary construction
- skip-gram pair generation
- negative sampling
- manual loss and gradients
- SGD updates

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

