An end-to-end movie recommendation system that covers data loading, preprocessing, content-based modeling, building a Streamlit web app, and deployment on Heroku. The system recommends similar movies based on the selected title, using features such as genres, keywords, cast, director, and overview.

<img width="737" height="454" alt="Screenshot 2025-09-26 at 21 08 26" src="https://github.com/user-attachments/assets/4bf78b2f-97fe-4a17-afb2-bcf0010a2265" />

## Highlights

- Content-based recommender system (no user history required)

- Posters fetched live via TMDb API

- NLP pipeline: parse JSON fields, clean, normalize, apply stemming, and vectorize with CountVectorizer

- Cosine similarity for top-N recommendations

- Streamlit-based UI with poster grid

- Ready for deployment on Heroku

## Tech Stack

`Python
pandas
numpy
scikit-learn (CountVectorizer)
nltk (PorterStemmer, stopwords)
Streamlit
requests
pickle
TMDb API
Heroku (optional)`

## Dataset

`TMDb 5000 Movies and TMDb 5000 Credits
Columns used: title, overview, genres, keywords, cast (top-3), crew (director), id`

`Note: JSON-like strings in genres, keywords, cast, and crew are parsed into lists. The id column is used to fetch posters.`

## Approach

- Merge movies.csv and credits.csv

- Select relevant columns: id, title, overview, genres, keywords, cast, crew

## Normalize fields:

- Parse JSON strings

- Extract top-3 actors and director

- Remove spaces in multi-word tokens

- Apply stemming to overview text

- Create tags column by combining all features

- Apply CountVectorizer with max 5000 features and English stopwords

- Compute cosine similarity matrix

- Recommend top-5 similar movies for a given title (with posters via TMDb)`

## Project Structure
 <img width="345" height="341" alt="image" src="https://github.com/user-attachments/assets/c4a55d1b-d423-4072-b92b-7bbe400a0302" />

Setup (Local)
`python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt`

`python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"`

##  Generate artifacts
`python - <<'PY'
from app.recommend import build_and_dump_artifacts
build_and_dump_artifacts(
    movies_csv='data/tmdb_5000_movies.csv',
    credits_csv='data/tmdb_5000_credits.csv',
    out_movies_dict='app/artifacts/movies_dict.pkl',
    out_similarity='app/artifacts/similarity.pkl'
)
PY`

## Set TMDb API key
export TMDB_API_KEY="YOUR_TMDB_API_KEY"

##  Run app
streamlit run app/app.py

### Usage

### Get top-5 recommended movies with posters
