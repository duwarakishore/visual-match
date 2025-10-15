import numpy as np
from numpy.linalg import norm

def cosine_similarity(a, b):
    return np.dot(a, b) / (norm(a) * norm(b))

def get_top_similar(query_feature, features, paths, top_k=5):
    similarities = []
    for i in range(len(features)):
        score = cosine_similarity(query_feature, features[i])
        similarities.append((paths[i], score))
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:top_k]
