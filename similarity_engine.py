"""Generate sentence embeddings for the pairwise similarity exercise.

Install dependencies with:
    python -m pip install sentence-transformers numpy

The first run downloads the all-MiniLM-L6-v2 model.
"""

import numpy as np
from sentence_transformers import SentenceTransformer


def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    """Return cosine similarity for two vectors using NumPy operations only."""
    norm_product = np.linalg.norm(v1) * np.linalg.norm(v2)
    if norm_product == 0:
        raise ValueError("Cosine similarity is undefined for a zero-length vector.")

    return float(np.dot(v1, v2) / norm_product)


def pairwise_similarity_matrix(embeddings: np.ndarray) -> np.ndarray:
    """Compute all pairwise cosine similarities with vectorized matrix multiplication."""
    if embeddings.ndim != 2:
        raise ValueError("Embeddings must be a 2D array (one vector per row).")

    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("Cosine similarity is undefined for zero-length embeddings.")

    normalized_embeddings = embeddings / norms
    return normalized_embeddings @ normalized_embeddings.T


def top_k_pairs(
    similarity_matrix: np.ndarray, sentences: list[str], k: int = 3
) -> list[tuple[int, int, float]]:
    """Return the k highest-scoring unique sentence pairs (excluding the diagonal)."""
    if similarity_matrix.shape != (len(sentences), len(sentences)):
        raise ValueError("Matrix dimensions must match the number of sentences.")
    if k < 0:
        raise ValueError("k must be non-negative.")

    # Only inspect the upper triangle, which excludes self-matches and duplicates.
    row_indices, column_indices = np.triu_indices(len(sentences), k=1)
    scores = similarity_matrix[row_indices, column_indices]
    best_indices = np.argsort(scores)[::-1][:k]

    return [
        (int(row_indices[index]), int(column_indices[index]), float(scores[index]))
        for index in best_indices
    ]


SENTENCES = [
    "A dog is playing fetch in the park.",
    "A puppy runs outside to catch a ball.",
    "The cat is sleeping on a warm sofa.",
    "A kitten naps on the couch in the sun.",
    "The chef prepared a pasta dinner for the family.",
    "A cook made noodles for an evening meal.",
    "The spacecraft launched toward Mars on Tuesday.",
    "Heavy rain flooded streets across the city.",
    "The violinist performed a classical concert.",
    "A new software update fixed several security bugs.",
]


def main() -> None:
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    embeddings = model.encode(SENTENCES, convert_to_numpy=True)

    print(f"Generated embeddings with shape: {embeddings.shape}")
    for index, (sentence, vector) in enumerate(zip(SENTENCES, embeddings), start=1):
        print(f"{index}. {sentence}  ->  {vector.shape[0]} dimensions")

    # Keep the generated vectors for the next steps of the exercise.
    np.save("sentence_embeddings.npy", embeddings)
    print("Saved vectors to sentence_embeddings.npy")

    # Example: compare the first two (intentionally similar) sentences.
    score = cosine_similarity(embeddings[0], embeddings[1])
    print(f"\nCosine similarity for sentences 1 and 2: {score:.4f}")

    similarity_matrix = pairwise_similarity_matrix(embeddings)
    print("\n10 x 10 pairwise cosine similarity matrix:")
    print(np.array2string(similarity_matrix, precision=3, suppress_small=True))

    print("\nTop 3 most similar sentence pairs:")
    for rank, (first, second, pair_score) in enumerate(
        top_k_pairs(similarity_matrix, SENTENCES, k=3), start=1
    ):
        print(f"{rank}. Score: {pair_score:.4f}")
        print(f"   Sentence {first + 1}: {SENTENCES[first]}")
        print(f"   Sentence {second + 1}: {SENTENCES[second]}")



if __name__ == "__main__":
    main()
