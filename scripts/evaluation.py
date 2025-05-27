import argparse
import os
import random
from datasets import Dataset
from dotenv import load_dotenv
import openai

from utils import load_faiss_index, load_json, load_model
from search_articles import query_index, select_top_articles_and_images, rerank_results
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision,
    answer_correctness,
    answer_similarity
)

def main(args):
    # load my key from .env file
    load_dotenv()
    openai.api_key = os.getenv("OPENAI_API_KEY")

    # load model and index
    model = load_model()
    index = load_faiss_index(args.index_path)
    metadata = load_json(args.meta_path)

    # choose random articles for test questions
    random.seed(42)
    text_articles = [entry for entry in metadata if entry["type"] == "text"]
    sample_articles = random.sample(text_articles, args.num_samples)
    contexts = [entry["raw_text"] for entry in sample_articles]

    # prompt template for GPT-based question generation
    qa_template = """As a research scientist testing comprehension of cutting-edge material,
    write a question that probes deep understanding of the context. Avoid superficial or general questions.
    Format:
    *Question*: question
    *Answer*: answer
    context: {context}
    """

    client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    synthetic_questions, synthetic_ground_truths = [], []

    for context in contexts:
        prompt = qa_template.format(context=context)
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=400,
        )
        result = response.choices[0].message.content
        question = result.split("*Question*:")[1].split("*Answer*:")[0].strip()
        answer = result.split("*Answer*:")[1].strip()
        synthetic_questions.append(question)
        synthetic_ground_truths.append(answer)

    # run retrieval and reranking
    retrieved_answers = []
    contexts_found = []

    for question in synthetic_questions:
        D, I = query_index(question, pool_size=args.pool_size, model=model, index=index)
        text_results, _ = select_top_articles_and_images(D, I, metadata, top_articles=10, top_images=0)
        text_results = rerank_results(question, text_results, model, top_k=1)

        if text_results:
            retrieved_answers.append(text_results[0][1]["raw_text"])
            contexts_found.append([t[1]["raw_text"] for t in text_results])
        else:
            retrieved_answers.append("")
            contexts_found.append([])

    data = {
        "question": synthetic_questions,
        "answer": retrieved_answers,
        "reference": contexts,
        "contexts": contexts_found,
        "ground_truth": synthetic_ground_truths,
    }
    dataset = Dataset.from_dict(data)
    # evaluate the dataset using RAGAS metrics
    result = evaluate(
        dataset=dataset,
        metrics=[
            context_precision,
            faithfulness,
            answer_relevancy,
            context_recall,
            answer_correctness,
            answer_similarity,
        ]
    )

    df_ragas = result.to_pandas()
    print(df_ragas)
    df_ragas.to_csv(args.output_csv, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAG Evaluation CLI with synthetic questions")
    parser.add_argument("--index_path", required=True, help="Path to FAISS index")
    parser.add_argument("--meta_path", required=True, help="Path to metadata JSON")
    parser.add_argument("--output_csv", default="ragas_evaluation_results.csv", help="Path to output CSV")
    parser.add_argument("--num_samples", type=int, default=10, help="Number of articles to sample")
    parser.add_argument("--pool_size", type=int, default=20, help="FAISS search pool size")

    args = parser.parse_args()
    main(args)
