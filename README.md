# 🎬 AI Movie Recommender

An intelligent movie recommendation system that uses **semantic search, emotion-based filtering, and genre selection** to recommend movies based on natural language queries.

The system uses **vector embeddings and a vector database** to understand the meaning of movie descriptions and recommend relevant movies.

---

## 🚀 Features

• 🔎 **Semantic Movie Search**  
Users can describe a movie in natural language and get similar movie recommendations.

• 🎭 **Emotion-Based Filtering**  
Movies can be filtered by emotional tone such as:
- Happy
- Sad
- Suspenseful
- Surprising
- Angry
- Disturbing
- Neutral

• 🎬 **Genre Filtering**  
Users can filter recommendations based on movie genre.

• 📊 **Large Dataset Support**  
Works with a dataset of **68,000+ movies**.

• 🧠 **Vector Embeddings**  
Movie descriptions are converted into embeddings using **Sentence Transformers**.

• ⚡ **Fast Semantic Search**  
Uses **Chroma vector database** for efficient similarity search.

• 🌐 **Interactive Dashboard**  
A **Gradio web interface** allows users to interact with the recommender system easily.

---

## 📊 Source Dataset

The project utilizes the **[TMDB Movies Dataset (2024)](https://www.kaggle.com/datasets/asaniczka/tmdb-movies-dataset-2023-930k-movies)**, a massive collection of 1,000,000 movies.

The Movie Database (TMDB) is a comprehensive database providing rich metadata, including titles, ratings, release dates, revenue, and genres, which serves as the backbone for our semantic search and filtering logic.

---

## 🤖 AI Models Used

This project leverages three specialized models from the HuggingFace ecosystem:

* **Embeddings**: `sentence-transformers/all-MiniLM-L6-v2`
    * *Role*: Converts plot summaries into 384-dimensional vectors for similarity search.
* **Zero-Shot Classifier**: `facebook/bart-large-mnli`
    * *Role*: Standardizes genre labels without requiring specific training on our dataset.
* **Emotion Classifier**: `j-hartmann/emotion-english-distilroberta-base`
    * *Role*: Scores the emotional tone of descriptions to enable mood-based ranking.

---

## 🛠 Architecture & Workflow

The system follows a modern RAG-inspired pipeline to process and retrieve recommendations:

1.  **Data Ingestion**: Loading movie metadata from CSV files.
2.  **Embedding Generation**: Using `sentence-transformers` to turn plot summaries into high-dimensional vectors.
3.  **Vector Storage**: Persisting embeddings in **ChromaDB** for efficient similarity lookups.
4.  **Retrieval**: Performing **Semantic Similarity Search** based on user natural language queries.
5.  **Post-Processing**: Applying **Genre + Emotion Filtering** to refine the search results.
6.  **User Interface**: Serving recommendations through an interactive **Gradio Dashboard**.


This project was developed in a sequential pipeline where each stage builds the necessary components for a high-performance recommendation engine.

### 1️⃣ Data Foundation (`movies-eda.ipynb`)
The process begins with **Exploratory Data Analysis**. This stage involves inspecting the raw TMDB dataset, handling missing values, and cleaning metadata. We analyze features like popularity and vote averages to prepare a high-quality dataset for the AI pipeline.

### 2️⃣ Semantic Search Engine (`vector-searcher.ipynb`)
Before adding filters, we build the "brain" of the system. We use **Sentence Transformers** to convert movie overviews into dense vector embeddings and store them in a **Chroma DB**. This enables the system to understand natural language queries like "dark space odyssey" based on meaning rather than just exact keywords.

### 3️⃣ Genre Standardization (`text-classification.ipynb`)
To improve filtering accuracy, we use **Zero-Shot Classification** to simplify the complex genre structures found in the raw data. This maps multi-genre tags into clean, standardized categories like "Action Science Fiction" or "Crime Thriller."

### 4️⃣ Tone & Emotion Analysis (`sentiment_analysis.ipynb`)
We further enrich the dataset by extracting the emotional "vibe" of each movie. Using a transformer-based classifier, we assign scores across seven emotions (e.g., Joy, Sadness, Fear), allowing users to filter recommendations based on their current mood.

### 5️⃣ Final Deployment (`recommender_dashboard.py`)
All components culminate in an interactive **Gradio Dashboard**. The application integrates the Chroma vector store and enriched metadata, allowing users to perform semantic searches, filter by genre, and rank by emotional tone in a seamless web interface.

---

## 📂 Project Structure

```text
AI-Movie-Recommender
│
├── data/
|   ├── movies.csv                # Main data used in initial codes
|   ├── final_movies_dataset.csv  # Cleaned and processed source data
│   ├── movies_complete.csv       # Preprocessed movie metadata
│   └── movies_with_genres.csv    # Dataset with specialized genre indexing
│
├── movie_vectors/                # Persistent ChromaDB collection files
│
├── recommender_dashboard.py      # Main entry point for the Gradio UI
│
├── movies-eda.ipynb              # Data Analysis and Cleaning
├── sentiment_analysis.ipynb      # Prototyping emotion detection logic
├── text-classification.ipynb     # Logic for genre/category classification
├── vector-searcher.ipynb         # Testing similarity thresholds and top-k retrieval
│
├── tagged_description.txt        # Documentation for tagging logic
│
├── requirements.txt              # Project dependencies
└── README.md                     # Project documentation
