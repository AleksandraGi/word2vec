# Minimal Word2Vec in NumPy
This is a minimal implementation of the core word2vec training loop in pure NumPy, using skip-gram with negative sampling. The focus is on implementing the forward pass, loss, gradients, and parameter updates from scratch rather than building a production-ready training pipeline.

The skip-gram variant with negative sampling were implemenmted. The goal was to understand and reproduce the forward pass, loss, gradients, and parameter updates from scratch, without using PyTorch or TensorFlow.

## Project goal
I wanted to reproduce the core idea of word2vec without using PyTorch, TensorFlow, or other machine learning frameworks.

## Model overview
This implementation uses:
- **Skip-gram**: for each center word, the model predicts words from its local context window
- **Negative sampling**: for each positive `(center, context)` pair, the model also sees several randomly sampled negative words

The model learns two embedding matrices:
- `W_in` for center words
- `W_out` for context words

## What the code does
The program:
1. reads a text from `sample_text.txt`
2. tokenizes the text
3. builds a vocabulary and maps words to integer ids
4. generates skip-gram training pairs
5. builds a negative sampling distribution from word frequencies
6. trains the embeddings using manual forward pass, loss, gradients, and SGD updates
7. prints the average loss per epoch
8. prints nearest neighbors for a few example words after training

## Files
- `main.py` - training loop and program entry point
- `preprocess.py` - text preprocessing, vocabulary building, pair generation, and negative sampling utilities
- `model.py` - the Word2Vec model, training step, and nearest-neighbor functions
- `sample_text.txt` - training text
- `sample_text2.txt` - second training text
- `README.md` - project description
- `note.md` - short description

## Output
During training, the program prints the average loss for each epoch. After training, it prints nearest neighbors for a few example words as a simple qualitative check of the learned embeddings.

The program prints the average loss per epoch, for example:
Epoch 1: loss = 2.7725
Epoch 2: loss = 2.7724
Epoch 3: loss = 2.7722
Epoch 4: loss = 2.7718
Epoch 5: loss = 2.7711

This value is the mean training loss over all positive pairs processed in a given epoch. In general, a decreasing loss suggests that the model is learning to:
- increase the scores of real (center, context) pairs
- decrease the scores of negative samples

After training, the program prints nearest neighbors for a few example words, for example:
Nearest neighbors for 'cat':
the          0.5550
rules        0.5313
on           0.4863
animals      0.3507
dog          0.3231

The exact output will depend on:
- the training text
- vocabulary size
- parameters
- random initialization
- sampled negatives during training

Because this is a small educational implementation, the nearest neighbors may not always look perfect, especially on a very small corpus. Still, they provide a useful sanity check that the embeddings are learning some structure from the data.


## Next Steps
Future work will include comparing this method with implementations built using PyTorch and TensorFlow.