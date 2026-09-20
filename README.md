# MoodShelf — Semantic Book Recommender

A full-stack NLP recommendation engine that lets you search a catalog of **5,197 books** using natural language ("a story about forgiveness and family") instead of keywords, then filter results by category and emotional tone.

![Python](https://img.shields.io/badge/Python-3.14-blue)
![LangChain](https://img.shields.io/badge/LangChain-ChromaDB-green)
![Gradio](https://img.shields.io/badge/UI-Gradio-orange)

## Features

- **Semantic search** over book descriptions using vector embeddings (Ollama `nomic-embed-text`) stored in **ChromaDB**, so queries match by meaning rather than exact keywords.
- **Automatic category classification** — raw Google Books categories are consolidated into 4 simple classes (Fiction, Nonfiction, Children's Fiction, Children's Nonfiction) via a rule-based mapping backed by a **zero-shot classifier** (`facebook/bart-large-mnli`) for uncatalogued books.
- **Emotion-aware re-ranking** — every description is scored across 7 emotions (joy, sadness, fear, anger, disgust, surprise, neutral) with a fine-tuned **DistilRoBERTa** model, powering tone filters like Happy, Sad, Suspenseful, Angry, and Surprising.
- **Interactive web app** built with Gradio: enter a query, optionally filter by category and tone, and get a gallery of the top 16 matches with covers and short blurbs.

## Tech Stack

| Layer | Tools |
|---|---|
| Data processing | Python, pandas |
| Category classification | Hugging Face Transformers (`facebook/bart-large-mnli`) |
| Emotion analysis | Hugging Face Transformers (`j-hartmann/emotion-english-distilroberta-base`) |
| Embeddings & vector search | LangChain, ChromaDB, Ollama (`nomic-embed-text`) |
| Web UI | Gradio |

## Pipeline

```
01_data_exploration.ipynb      → clean raw book metadata
02_zero_shot_category.ipynb    → map + zero-shot classify into simple_categories
03_sentiment_analysis.ipynb    → score emotions per book description
04_vector_search.ipynb         → build the ChromaDB vector index
app.py                         → Gradio app tying it all together
```

Each notebook reads the previous notebook's output CSV and writes its own, so they're meant to be run **in order** the first time.

## Project Structure

```
.
├── app.py                     # Gradio front-end
├── requirements.txt
├── .env                       # not committed — see Setup
├── assets/
│   └── cover-not-found.jpg    # placeholder cover image
├── data/
│   ├── books_cleaned.csv
│   ├── books_with_categories.csv
│   └── books_with_emotions.csv
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_zero_shot_category.ipynb
│   ├── 03_sentiment_analysis.ipynb
│   └── 04_vector_search.ipynb
└── chroma_db/                 # generated vector index (not committed)
```

## Setup

**Prerequisites:** Python 3.10+, [Ollama](https://ollama.com) installed and running locally.

```bash
# 1. Clone the repo
git clone https://github.com/Isha-NIT/MoodShelf--book-recommendation-system.git
cd MoodShelf--book-recommendation-system

# 2. Create a virtual environment
python -m venv myenv
myenv\Scripts\activate       # Windows
source myenv/bin/activate    # macOS/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Pull the embedding model
ollama pull nomic-embed-text
```

## Running the Pipeline

If `data/books_with_emotions.csv` and `chroma_db/` aren't already present, run the notebooks in order (`01` → `04`) to regenerate them. Otherwise, skip straight to the app.

```bash
python app.py
```

Then open the local URL Gradio prints in your terminal.

## License

[MIT](LICENSE) — or your license of choice.
