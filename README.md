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

## Next Steps
Future work will include comparing this method with implementations built using PyTorch and TensorFlow.