# 🔍 Multimodal Retrieval-Augmented Generation System (Text + Images)
👉 [Demo](https://sofibrezden-multimodal-rag-appapp-s372hk.streamlit.app/)

This project is a full-stack implementation of a multimodal RAG (Retrieval-Augmented Generation) system that combines
textual and visual data to retrieve context-relevant information and generate coherent answers using LLMs. The app
supports both article and image-based retrieval, with evaluation based on RAGAS metrics.

## 📚 Table of Contents

- [🚀 Features](#-features)
- [📁 Project Structure](#-project-structure)
- [⚙️ How It Works](#-how-it-works)
- [🛠️ Installation](#-installation)
- [🔑 OpenAI API Key Setup](#-setup-OpenAI-Key)
- [🧪 Running the Components](#-running-the-components)
- [🧠 RAGAS Evaluation](#-ragas-evaluation)
- [📈 Possible Improvements](#-possible-improvements)
- [🌐 Live Demo](#-live-demo)

---

## 🚀 Features

- 🔎 FAISS-powered vector search over text and BLIP-generated image captions
- 🤖 OpenAI LLM (`gpt-4o`) for multimodal question answering
- 📸 BLIP-based captioning of images from articles
- 🧠 Semantic reranking using `sentence-transformers`
- 📊 RAGAS evaluation of retrieval and answer quality
- 🧾 Streamlit-based interactive UI
- 📁 Custom filters: View only articles, images, or both
- 🔐 API key management via `.env` or `secrets.toml`

---

## 📁 Project Structure

```
project/
├── app/
│ └── app.py # Streamlit app UI
├── scripts/
│ ├── build_index.py # Index building script (text + images)
│ ├── evaluation.py # RAGAS evaluation pipeline
│ ├── scrap_articles.py # Web scraper for article data
│ ├── search_articles.py # Retrieval logic and reranker
│ └── utils.py # Loaders for model, index, metadata
├── output/
│ ├── *.index, *.json # FAISS index and metadata
│ └── the_batch_articles.jsonl # Scraped articles in JSONL format
├── report.pdf # report on the system design and evaluation
├── demo_rag.gif # GIF demo of the app
├── requirements.txt
└── README.md
```

## ⚙️ How It Works

1. **Web Scraping**: Using Selenium and BeautifulSoup, articles are scraped
   from [deeplearning.ai/the-batch](https://www.deeplearning.ai/the-batch/), extracting titles, dates, text, images, and
   URLs.
2. **Image Captioning**: Each article image is summarized using the BLIP model to generate a descriptive caption.
3. **Vector Indexing**: Both text and image captions are embedded using Sentence Transformers and stored in a FAISS
   index.
4. **Retrieval & Reranking**: Queries retrieve top documents by vector similarity and are then reranked using semantic
   similarity to the query.
5. **Question Answering (LLM)**: Retrieved context (text + captions) is fed to OpenAI GPT-4o for response generation.
6. **Evaluation**: System responses are scored using RAGAS metrics for answer quality and context relevance.
7. **Frontend**: A Streamlit UI allows users to explore results, filter by type, and compare retrieval vs LLM answers.

### Prerequisites

- Python 3.8+
- `pip` for package management
- OpenAI API key (stored in `.env` and `secrets.toml`)

## ⚙️ Installation

```bash
# Clone the repository
git clone https://github.com/sofibrezden/multimodal_rag.git
cd multimodal-rag

# Set up virtual environment
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt
```

## 🔑 Setup OpenAI Key

Add your OpenAI API key in one of two ways:

### Option 1: Using .env (for scripts)

#### .env

```bash
OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

### Option 2: Using secrets.toml (for Streamlit UI)

##### .streamlit/secrets.toml

```
OPENAI_API_KEY = "sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
```

# 🧪 Running the Components

#### 1. Scrape Articles

```bash
python scripts/scrap_articles.py
```

#### 2. Build the FAISS Index

```bash
python scripts/build_index.py \
    --input output/the_batch_articles.jsonl \
    --index_path output/textonly_blip.index \
    --meta_path output/textonly_meta_blip.json
```

#### 3.📊 Evaluate the System

```bash
python scripts/evaluation.py
```

#### 4. Start the Streamlit App

```bash
streamlit run app/app.py
```

Your app will be available at `http://localhost:8501`.

# 🧠 RAGAS Evaluation

The system uses RAGAS for evaluation with metrics like:

* Context Precision
* Context Recall
* Answer Relevancy
* Faithfulness
* Answer Similarity

Evaluation results are saved to ```output/ragas_evaluation_results.csv```.

# 📈 Possible Improvements

* Replace BLIP with BLIP-2 for better captions
* Use ColBERT or bge-reranker for deeper reranking
* Enable hybrid (image + text) retrieval in parallel
* Add document chunking for long articles
* Support user uploads for image-based questions

## 🌐 Live Demo

You can try the app here:  
👉 [Demo](https://sofibrezden-multimodal-rag-appapp-s372hk.streamlit.app/)

![Demo GIF](demo_rag.gif)