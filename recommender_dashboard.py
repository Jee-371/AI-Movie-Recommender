import pandas as pd
from transformers import pipeline
import gradio as gr
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import os

movies = pd.read_csv(r'data\movies_complete.csv')

raw_documents = TextLoader(
    "tagged_description.txt",
    encoding="utf-8"
).load()

text_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=1,
    chunk_overlap=0
)

documents = text_splitter.split_documents(raw_documents)

embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

if os.path.exists("movie_vectors"):
    db_movies = Chroma(
        persist_directory="movie_vectors",
        embedding_function=embedding
    )
else:
    db_movies = Chroma.from_documents(
        documents,
        embedding,
        persist_directory="movie_vectors"
    )


def retrieve_semantic_recommendations(
        query: str,
        category: str = None,
        tone: str = None,
        initial_top_k: int = 50,
        final_top_k: int = 16
) -> pd.DataFrame:

    recs = db_movies.similarity_search_with_score(query, k=initial_top_k)

    movie_ids = [
        int(rec[0].page_content.split("MOVIE_ID:")[1].split(",")[0].strip())
        for rec in recs
    ]

    movie_recs = movies[movies["id"].isin(movie_ids)].head(final_top_k)

    # Genre filtering
    if category and category != "All":
        movie_recs = movie_recs[
            movie_recs["genre_category"] == category
        ].head(final_top_k)

    # Emotion sorting
    if tone == "Happy":
        movie_recs.sort_values(by="joy", ascending=False, inplace=True)

    elif tone == "Sad":
        movie_recs.sort_values(by="sadness", ascending=False, inplace=True)

    elif tone == "Angry":
        movie_recs.sort_values(by="anger", ascending=False, inplace=True)

    elif tone == "Suspenseful":
        movie_recs.sort_values(by="fear", ascending=False, inplace=True)

    elif tone == "Surprising":
        movie_recs.sort_values(by="surprise", ascending=False, inplace=True)

    elif tone == "Disturbing":
        movie_recs.sort_values(by="disgust", ascending=False, inplace=True)

    return movie_recs.head(final_top_k)

def recommend_movies(query, category, tone):

    results = retrieve_semantic_recommendations(
        query=query,
        category=category,
        tone=tone
    )

    if results.empty:
        return "No movies found."

    output = ""

    for _, row in results.iterrows():

        output += f"""
🎬 {row['title']}

        ⭐ Rating: {row['vote_average']}
        🎭 Genre: {row['genre_category']}
        ⏱ Runtime: {row['runtime']} minutes

        📖 Overview:
        {row['overview']}

        ------------------------------------
"""

    return output

categories = ['All'] + sorted(movies['genre_category'].unique())
tones = ['All'] + ['Happy', 'Sad', 'Surprising', 'Suspenseful', 'Angry', 'Disturbing']

def gradio_recommend(query, category, tone):

    if category == "All":
        category = None

    if tone == "All":
        tone = None

    results = retrieve_semantic_recommendations(
        query=query,
        category=category,
        tone=tone
    )

    if results.empty:
        return "No movies found."

    output = ""

    for _, row in results.iterrows():

        output += f"""
<div class="movie-card">

<h2>{row['title']}</h2>

<b>⭐ Rating:</b> {row['vote_average']}  
<b>🎭 Genre:</b> {row['genre_category']}  
<b>⏱ Runtime:</b> {row['runtime']} minutes  

<p>{row['overview']}</p>

</div>
"""

    return output


css = """

body {
background: linear-gradient(135deg,#1e1e2f,#121212);
color:white;
font-family: Arial;
}

.movie-card{
background: rgba(255,255,255,0.08);
backdrop-filter: blur(12px);
border-radius:15px;
padding:20px;
margin-bottom:15px;
box-shadow:0 8px 30px rgba(0,0,0,0.4);
}

"""


with gr.Blocks(css=css) as demo:

    gr.Markdown(
        """
        # 🎬 AI Movie Recommender  
        Describe a movie and get smart recommendations
        """
    )

    with gr.Row():

        query = gr.Textbox(
            label="Movie Description",
            placeholder="example: funny space adventure or crime thriller"
        )

    with gr.Row():

        category = gr.Dropdown(
            choices=categories,
            value="All",
            label="Genre"
        )

        tone = gr.Dropdown(
            choices=tones,
            value="All",
            label="Emotion / Tone"
        )

    recommend_btn = gr.Button("Recommend Movies")

    output = gr.HTML()

    recommend_btn.click(
        fn=gradio_recommend,
        inputs=[query, category, tone],
        outputs=output
    )

demo.launch()