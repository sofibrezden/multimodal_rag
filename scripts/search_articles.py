import argparse
import numpy as np
from sentence_transformers import util
import faiss
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from scripts.utils import load_model, load_jsonl


def rerank_results(query, results, model, top_k=5):
    # re-rank search results using cosine similarity between the query and the candidate documents
    if not results:
        return []
    query_emb = model.encode(query, convert_to_tensor=True)
    texts = [entry['raw_text'] for _, entry, _ in results]
    doc_embs = model.encode(texts, convert_to_tensor=True)

    cos_scores = util.pytorch_cos_sim(query_emb, doc_embs)[0]
    ranked = sorted(zip(results, cos_scores), key=lambda x: x[1], reverse=True)

    return [r[0] for r in ranked[:top_k]]


def query_index(query, pool_size, model, index):
    # perform a FAISS vector search on the index using the encoded query
    query_emb = model.encode([query], convert_to_numpy=True)
    D, I = index.search(np.array(query_emb).astype(np.float32), pool_size)
    return D[0], I[0]


def select_top_articles_and_images(D, I, metadata, top_articles=5, top_images=3):
    # filter and select top ranked text and image results based on type
    sorted_results = [(idx, metadata[idx], D[i]) for i, idx in enumerate(I)]
    text_results = [r for r in sorted_results if r[1]['type'] == 'text']
    image_results = [r for r in sorted_results if r[1]['type'] == 'image']
    return text_results[:top_articles], image_results[:top_images]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run FAISS index query and reranking")
    parser.add_argument("--query", required=True, help="Query string to search")
    parser.add_argument("--top_k", type=int, default=5, help="Number of top reranked results")
    parser.add_argument("--pool_size", type=int, default=20, help="Pool size from FAISS search")
    parser.add_argument("--index_path", required=True, help="Path to FAISS index")
    parser.add_argument("--meta_path", required=True, help="Path to JSON metadata")
    args = parser.parse_args()

    model = load_model("all-MiniLM-L6-v2")
    index = faiss.read_index(args.index_path)
    metadata = load_json(args.meta_path)

    D, I = query_index(args.query, args.pool_size, model, index)
    text_results, image_results = select_top_articles_and_images(D, I, metadata)

    top_results = rerank_results(args.query, text_results, model, top_k=args.top_k)

    print("\nTop reranked text results:")
    for idx, entry, _ in top_results:
        print(f"- {entry['raw_text'][:200]}...")

    print("\nTop image URLs:")
    for idx, entry, _ in image_results[:args.top_k]:
        print(f"- {entry['meta'].get('image_url', 'No URL')}")
