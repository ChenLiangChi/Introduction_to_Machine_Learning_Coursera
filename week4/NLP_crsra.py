import h5py
import numpy as np

with h5py.File('/Users/liangchichen/Desktop/intro_to_ML/week4/dataset/mini.h5', 'r') as f:
    all_words = [word.decode('utf-8') for word in f['mat']['axis1'][:]]
    all_embeddings = f['mat']['block0_values'][:]
    
    english_words = [word[6:] for word in all_words if word.startswith('/c/en/')]
    english_word_indices = [i for i, word in enumerate(all_words) if word.startswith('/c/en/')]
    english_embeddings = all_embeddings[english_word_indices]

# L2 normalized
norms = np.linalg.norm(english_embeddings, axis=1)
normalized_embeddings = english_embeddings.astype('float32') / norms.astype('float32').reshape([-1, 1])

index = {word: i for i, word in enumerate(english_words)}

def similarity_score(w1, w2):
    score = np.dot(normalized_embeddings[index[w1], :], normalized_embeddings[index[w2], :])
    return score

def closest_to_vector(v, n):
    all_scores = np.dot(normalized_embeddings, v)
    best_words = map(lambda i: english_words[i], reversed(np.argsort(all_scores)))
    return [next(best_words) for _ in range(n)]

def most_similar(w, n):
    return closest_to_vector(normalized_embeddings[index[w], :], n)


print(most_similar('cat', 10))
print(most_similar('dog', 10))
print(most_similar('duke', 10))
