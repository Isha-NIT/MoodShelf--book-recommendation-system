"""Gradio front-end for MoodShelf, the semantic book recommender.

Run from the project root:  python app.py
Requires: Ollama running with `nomic-embed-text`.
"""

from pathlib import Path

import gradio as gr
import numpy as np
import pandas as pd
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

BASE_DIR = Path(__file__).resolve().parent
COVER_PLACEHOLDER = str(BASE_DIR / "assets" / "cover-not-found.jpg")

# Dropdown label -> emotion column used to re-rank results.
TONE_TO_EMOTION = {
    "Happy": "joy",
    "Surprising": "surprise",
    "Angry": "anger",
    "Suspenseful": "fear",
    "Sad": "sadness",
}


# Data and vector store
books = pd.read_csv(BASE_DIR / "data" / "books_with_emotions.csv")
books["large_thumbnail"] = np.where(
    books["thumbnail"].isna(),
    COVER_PLACEHOLDER,
    books["thumbnail"] + "&fife=w800", 
)

db_books = Chroma(
    collection_name="book_descriptions",
    persist_directory=str(BASE_DIR / "chroma_db"),
    embedding_function=OllamaEmbeddings(model="nomic-embed-text"),
    collection_metadata={"hnsw:space": "cosine"},
)



# Recommendation logic
def retrieve_semantic_recommendations(
    query: str,
    category: str = "All",
    tone: str = "All",
    initial_top_k: int = 50,
    final_top_k: int = 16,
) -> pd.DataFrame:
    """Return up to `final_top_k` books for `query`.

    Steps: semantic search (top `initial_top_k`, best match first) ->
    optional category filter -> optional tone re-rank over the whole
    filtered pool -> keep the first `final_top_k`.
    """
    category = category or "All"
    tone = tone or "All"

    docs = db_books.similarity_search(query, k=initial_top_k)
    ranked_isbns = [int(doc.metadata["isbn13"]) for doc in docs]
    rank_of = {isbn: rank for rank, isbn in enumerate(ranked_isbns)}

    recs = books[books["isbn13"].isin(rank_of)].copy()
    recs["similarity_rank"] = recs["isbn13"].map(rank_of)
    recs = recs.sort_values("similarity_rank")  

    if category != "All":
        recs = recs[recs["simple_categories"] == category]
    if tone != "All":
        recs = recs.sort_values(TONE_TO_EMOTION[tone], ascending=False)

    return recs.head(final_top_k)


def format_authors(raw_authors) -> str:
    """Turn 'A;B;C' into 'A, B, and C' (handles missing authors)."""
    if pd.isna(raw_authors):
        return "Unknown author"
    names = raw_authors.split(";")
    if len(names) == 1:
        return names[0]
    if len(names) == 2:
        return f"{names[0]} and {names[1]}"
    return f"{', '.join(names[:-1])}, and {names[-1]}"


def recommend_books(query: str, category: str, tone: str) -> list[tuple[str, str]]:
    """Gradio callback: return (image, caption) pairs for the gallery."""
    if not query or not query.strip():
        gr.Warning("Please enter a description of a book.")
        return []

    recommendations = retrieve_semantic_recommendations(query, category, tone)
    results = []
    for _, row in recommendations.iterrows():
        short_description = " ".join(row["description"].split()[:30]) + "..."
        caption = f"{row['title']} by {format_authors(row['authors'])}: {short_description}"
        results.append((row["large_thumbnail"], caption))
    return results



# UI
categories = ["All"] + sorted(books["simple_categories"].dropna().unique())
tones = ["All", *TONE_TO_EMOTION]

with gr.Blocks(theme=gr.themes.Glass()) as dashboard:
    gr.Markdown("# Semantic book recommender")

    with gr.Row():
        user_query = gr.Textbox(
            label="Please enter a description of a book:",
            placeholder="e.g., A story about forgiveness",
        )
        category_dropdown = gr.Dropdown(
            choices=categories, label="Select a category:", value="All"
        )
        tone_dropdown = gr.Dropdown(
            choices=tones, label="Select an emotional tone:", value="All"
        )
        submit_button = gr.Button("Find recommendations")

    gr.Markdown("## Recommendations")
    output = gr.Gallery(label="Recommended books", columns=8, rows=2)

    submit_button.click(
        fn=recommend_books,
        inputs=[user_query, category_dropdown, tone_dropdown],
        outputs=output,
    )


if __name__ == "__main__":
    dashboard.launch(allowed_paths=[str(BASE_DIR / "assets")])