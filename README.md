# 🎬 Content-Based Movie Recommendation System
### A production-ready NLP pipeline that drives personalized discovery without requiring any user history, ratings, or behavioral data.

> *A business-driven recommender built to solve cold-start, demonstrate NLP feature engineering, and ship a live web app backed by real-time poster fetching via the TMDb API.*

---

## 💼 Business Problem

Streaming platforms and movie databases live or die by one metric: **time-to-next-watch**. When a user finishes a film and finds nothing compelling to play next, they churn. When discovery feels effortless, they stay.

The core challenge: most recommendation systems require a history of user ratings or behavioral signals to function. New users, niche titles, and cold-start scenarios have no such data. A platform needs recommendations it can serve *immediately*, from content alone, with zero prior interaction.

**The question this project answers:**
> *Can we build a recommendation engine that finds genuinely similar movies purely from what a film is about, who made it, and who stars in it - without needing any user data at all?*

---

## ❗ Why This Matters

Content-based filtering is not just a fallback for when collaborative filtering fails. It is strategically valuable in its own right:

- **Cold start is a real revenue problem.** Every new user who gets bad recommendations in their first session is a potential cancellation. Content-based models work from day one.
- **Catalog depth.** Streaming services host tens of thousands of titles. The long tail of less-watched films only gets surfaced if the system can match them to something the viewer just watched, without relying on other users having seen them first.
- **Transparency and explainability.** "We recommended this because it shares a director, genre, and cast with a film you just watched" is an answer a product team can explain to users and to regulators. Black-box collaborative models cannot do this as cleanly.
- **Privacy compliance.** Content-based models operate on metadata the platform already owns, without requiring user-level data storage or GDPR-sensitive behavioral tracking.

---

## 🎯 Objective

Build an end-to-end content-based recommendation pipeline that:

1. Merges and cleans two real-world TMDb datasets (movies + credits)
2. Engineers a unified "semantic fingerprint" for each film from five distinct feature types
3. Applies NLP preprocessing (stemming, stopword removal, space normalization) to improve token quality
4. Vectorizes the fingerprint with Bag-of-Words and computes pairwise cosine similarity across 4,806 films
5. Surfaces top-5 recommendations for any selected title, with live poster fetching via the TMDb API
6. Deploys as a usable Streamlit web application

---

## 💰 Business Impact

*Estimates are modeled against realistic streaming industry benchmarks to illustrate production-scale value.*

| Impact Area | Estimate | Basis |
|---|---|---|
| **Cold-start coverage** | 100% of catalog recommends from day one | No user history required |
| **Session extension** | +1 to 2 additional titles per session | Good recommendations add 15-25% watch time per industry benchmarks |
| **Churn reduction potential** | Meaningful reduction in first-session drop-off | Users who find a second title are significantly more likely to return |
| **Catalog utilization** | Long-tail titles surfaced by semantic match | Without content signals, niche films are invisible to new users |
| **Infrastructure cost** | Near-zero inference cost post-build | Similarity matrix computed once; lookup is O(1) at serve time |
| **Privacy compliance** | No user-level data required or stored | Fully GDPR-compatible by design |

> **Note:** This is a portfolio project using the public TMDb 5000 dataset. Business impact figures are modeled assumptions showing how this architecture scales in production, not claims from a live deployment.

---

## 📊 Dataset Overview

**Source:** [TMDb 5000 Movie Dataset - Kaggle](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata)

Two files merged on `id` / `movie_id`:

| File | Rows | Key Columns Used |
|---|---|---|
| `tmdb_5000_movies.csv` | 4,809 | `id`, `title`, `overview`, `genres`, `keywords` |
| `tmdb_5000_credits.csv` | 4,809 | `cast`, `crew` |

**After merge and null removal: 4,806 films**

Features used to build recommendations:

| Feature | Raw Format | Extracted |
|---|---|---|
| `genres` | JSON string | All genre names |
| `keywords` | JSON string | All keyword tags |
| `cast` | JSON string | Top 3 billed actors |
| `crew` | JSON string | Director only |
| `overview` | Plain text | Full plot description (tokenized + stemmed) |

---

## 🧠 Methodology and Decision Process

### Why content-based over collaborative filtering?

Collaborative filtering requires a dense user-item rating matrix. The TMDb dataset has no user ratings. More importantly, building a system that works with zero behavioral data is the more interesting engineering challenge and the more deployable real-world solution. Content-based filtering also generalizes naturally to any domain with rich metadata: books, podcasts, products, job listings.

### The "semantic fingerprint" approach

Each film is reduced to a single `tags` string that concatenates five feature types. This is a deliberate design choice: rather than running separate similarity computations per feature and weighting them, the concatenation approach lets the vectorizer naturally weight terms by frequency across the corpus. A genre like "Action" appearing in 800 films contributes less discriminating power per occurrence than a rare keyword like "dystopia" or a specific director's name.

### Why space removal on multi-word tokens?

Actor names and director names like "Sam Mendes" would be split by the vectorizer into `sam` and `mendes` - two meaningless tokens on their own. By collapsing them to `SamMendes`, the vectorizer treats the full name as a single, meaningful token. This is a small but consequential preprocessing step: without it, "Sam Raimi" and "Sam Mendes" look related because they share `sam`, and the similarity matrix degrades accordingly.

### Why stemming?

`loved`, `loving`, `loves`, and `love` are semantically identical but would be treated as four separate features by a raw CountVectorizer. PorterStemmer collapses them to a single root. This reduces the effective vocabulary and improves token matching across films with similar thematic content described in slightly different words.

### Why Bag-of-Words over TF-IDF?

TF-IDF would downweight tokens that appear frequently across the corpus, which sounds appealing but creates a problem here: genre tags like `Action` and `Drama` appear in hundreds of films, and TF-IDF would suppress them. For movie recommendations, a shared genre *is* meaningful signal. Bag-of-Words preserves the full weight of these high-frequency but semantically important shared tokens. TF-IDF would be more appropriate if the goal were keyword extraction or search, not similarity scoring.

### Why cosine similarity over Euclidean distance?

In high-dimensional vector spaces (5,000 features here), Euclidean distance is unreliable: longer documents get penalized purely because they have more words, not because they are less similar. Cosine similarity measures the angle between vectors, making it invariant to document length. Two films with the same genre, keyword, and cast profile but different overview lengths will still score high similarity, as they should.

### Tradeoffs Considered

- **Top-3 cast vs. full cast:** Including all cast members would drown out director and genre signals with minor supporting roles. Top-3 billing captures the actors most associated with a film's identity.
- **Director only from crew:** The crew JSON contains hundreds of entries (cinematographers, editors, composers). Only the director has consistent, reliable influence on a film's tone and style. Including other crew roles added noise without signal improvement.
- **max_features=5000:** A reasonable ceiling for a 4,806-film corpus. Beyond 5,000, the feature space starts to include rare one-off tokens with no discriminating power while increasing compute cost.

---

## 🔍 NLP Pipeline: Step by Step

```
Raw JSON strings (genres, keywords, cast, crew)
         |
   ast.literal_eval() - parse to Python lists
         |
   Extract names (top-3 cast, director only)
         |
   Remove spaces in multi-word tokens ("Sam Mendes" -> "SamMendes")
         |
   overview.split() - tokenize plot text
         |
   Concatenate all features into single 'tags' string
         |
   Lowercase normalization
         |
   PorterStemmer - reduce to root forms
         |
   CountVectorizer(max_features=5000, stop_words='english')
         |
   5000-dimensional Bag-of-Words vector per film
         |
   cosine_similarity() - 4806 x 4806 similarity matrix
         |
   recommend(title) -> top-5 similar films by cosine score
```

---

## 📈 Results

**Sample output for `recommend('Batman Begins')`:**

The system correctly surfaces films that share Christopher Nolan's direction, the superhero/action/crime genre cluster, and thematically related keywords - without any user data, ratings, or collaborative signal.

**Similarity matrix dimensions:** 4,806 x 4,806 (23.1 million pairwise scores)

**Vocabulary size:** 5,000 most frequent stems after stopword removal

**Inference speed:** O(1) lookup after one-time matrix computation - suitable for real-time serving

---

## 🌐 Web App and Deployment

The recommendation engine is wrapped in a Streamlit application that:

- Presents a dropdown of all 4,806 film titles
- Returns the top-5 recommended titles on selection
- Fetches and displays live movie posters for each recommendation via the TMDb API
- Requires only a free TMDb API key to run

The live poster fetching was a deliberate UX decision: text-only recommendations are harder to scan quickly. Posters give users an immediate visual context for whether a recommendation feels right, reducing friction in the discovery flow.

---

## ⚙️ Tech Stack

**Data:** Python, Pandas, NumPy, `ast` (JSON field parsing)

**NLP:** scikit-learn (CountVectorizer), NLTK (PorterStemmer, stopwords)

**Similarity:** scikit-learn (cosine_similarity)

**Web App:** Streamlit

**External API:** TMDb API (live poster fetching)

**Dataset:** TMDb 5000 Movies + TMDb 5000 Credits (Kaggle)

---

## 🤖 How AI Was Used

| Task | AI-Assisted? | My Role |
|---|---|---|
| Core pipeline architecture | No | Designed independently |
| JSON parsing approach (`ast.literal_eval`) | No | Standard Python pattern, applied independently |
| Space-removal normalization logic | No | My own design decision based on vectorizer behavior |
| BoW vs. TF-IDF tradeoff | ChatGPT consulted for vocabulary | Final choice and justification mine |
| README structure and business framing | ChatGPT suggested framing approach | All content, numbers, and analysis written by me |
| Streamlit app layout | No | Built independently |

**Principle:** AI was used to accelerate thinking on specific tradeoffs. Every architectural and analytical decision was made and validated independently.

---

## 🌍 How This Framework Applies Elsewhere

The content-based similarity pipeline built here is domain-agnostic. The same architecture transfers directly to:

- **E-commerce:** Replace `tags` with product category + brand + description + material. "Customers who viewed this" recommendations without any purchase history.
- **Job boards:** Combine job title + required skills + industry + seniority. Match candidates to roles by profile similarity, not just keyword search.
- **News platforms:** Use article headline + section + author + named entities. Surface related reading without needing click history.
- **Music:** Genre + artist + mood tags + era. Playlist generation from a single seed song.
- **Books / podcasts:** Genre + themes + author + keywords from description. Cold-start recommendations for new users with no interaction history.

The pipeline structure - parse structured metadata, engineer a unified text representation, vectorize, compute pairwise similarity, serve top-N - is reusable with a dataset swap and minor feature engineering adjustments.

---

## 📋 Step-by-Step Reproduction Guide

**1. Data Acquisition**

Download both files from [Kaggle](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata): `tmdb_5000_movies.csv` and `tmdb_5000_credits.csv`.

**2. Environment Setup**
```bash
pip install pandas numpy scikit-learn nltk requests streamlit
python -m nltk.downloader stopwords punkt
```

**3. Data Merge and Column Selection**
```python
movies_data = movies.merge(credits, left_on='id', right_on='movie_id')
movies_data = movies_data[['id', 'title_x', 'genres', 'keywords', 'overview', 'cast', 'crew']]
movies_data = movies_data.dropna().rename(columns={'title_x': 'title'})
```

**4. JSON Field Parsing**
```python
import ast

def convert(obj):
    return [i['name'] for i in ast.literal_eval(obj)]

def convert_3(obj):
    return [i['name'] for i in ast.literal_eval(obj)][:3]

def fetch_director(obj):
    return [i['name'] for i in ast.literal_eval(obj) if i['job'] == 'Director'][:1]

movies_data['genres'] = movies_data['genres'].apply(convert)
movies_data['keywords'] = movies_data['keywords'].apply(convert)
movies_data['cast'] = movies_data['cast'].apply(convert_3)
movies_data['crew'] = movies_data['crew'].apply(fetch_director)
```

**5. Space Removal (Multi-word Token Fix)**
```python
cols = ['genres', 'keywords', 'cast', 'crew']
for c in cols:
    movies_data[c] = movies_data[c].apply(lambda x: [i.replace(' ', '') for i in x])
```

**6. Tag Construction**
```python
movies_data['overview'] = movies_data['overview'].apply(lambda x: x.split())
movies_data['tags'] = (movies_data['overview'] + movies_data['genres'] +
                       movies_data['keywords'] + movies_data['cast'] + movies_data['crew'])

new_df = movies_data[['id', 'title', 'tags']].copy()
new_df['tags'] = new_df['tags'].apply(lambda x: ' '.join(x)).str.lower()
```

**7. Stemming**
```python
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()

def stem(text):
    return " ".join([ps.stem(word) for word in text.split()])

new_df['tags'] = new_df['tags'].apply(stem)
```

**8. Vectorization and Similarity Matrix**
```python
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(new_df['tags']).toarray()
similarity = cosine_similarity(vectors)
```

**9. Recommendation Function**
```python
def recommend(movie):
    idx = new_df[new_df['title'] == movie].index[0]
    distances = similarity[idx]
    movies_list = sorted(enumerate(distances), key=lambda x: x[1], reverse=True)[1:6]
    return [new_df.iloc[i[0]].title for i in movies_list]
```

**10. TMDb Poster Fetching**

Get a free API key at [themoviedb.org](https://www.themoviedb.org/settings/api). Then:
```python
import requests

def fetch_poster(movie_id, api_key):
    url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={api_key}"
    data = requests.get(url).json()
    return "https://image.tmdb.org/t/p/w500/" + data['poster_path']
```

**11. Run the App**
```bash
streamlit run app.py
```

---

## 📖 Context

**Project type:** Independent portfolio project  
**Role:** Solo end-to-end - data merging, NLP pipeline, similarity engine, Streamlit app, API integration  
**Stakeholder simulation:** Designed as a layer a streaming platform or movie discovery site could adopt for cold-start and catalog coverage  
**Constraints:** Public TMDb metadata only - no user ratings, no behavioral signals, no proprietary features

---

## 💡 Key Learnings

**What I would improve:**

- **TF-IDF hybrid:** Apply TF-IDF weighting only to the `overview` component (where word frequency variance matters) while keeping raw counts for structured tags like genres, cast, and director. This would give better signal balance between free text and categorical metadata.
- **Weighted feature concatenation:** Repeat high-signal tokens (e.g., director name 3x, genre 2x) to simulate feature weighting without rebuilding the pipeline. A more principled approach would use separate embeddings per feature type with learned weights.
- **Word embeddings:** Bag-of-Words cannot capture that "heist" and "robbery" are semantically related. A sentence-transformer embedding on the `overview` field would dramatically improve matching for thematically similar films that use different vocabulary.
- **Evaluation framework:** The current pipeline has no quantitative evaluation metric. Adding precision@k using a held-out set of known similar films (sequels, same-director pairs) would make model comparisons rigorous.

**What surprised me:**

- The space-removal step for multi-word tokens had a larger impact on recommendation quality than expected. Early tests without it produced spurious similarity between films that merely shared common first names in their cast lists.
- CountVectorizer's English stopword list removes words like "not" and "never", which can meaningfully change the semantic intent of an overview. For a recommendation system this is acceptable, but for sentiment-sensitive applications it would need explicit handling.
- The 4,806 x 4,806 similarity matrix (23 million float values) is fast to query but expensive to store and recompute. For a catalog of 50,000+ titles, this approach would require approximate nearest neighbor methods (e.g., FAISS) rather than full matrix computation.

**Business insight gained:**

The most important design decision in a content-based system is not the model - it is the feature set. A Bag-of-Words model on well-engineered tags consistently outperforms a more sophisticated model on poorly chosen features. The time spent on JSON parsing, token normalization, and stemming delivered more recommendation quality improvement than any modeling choice made after the pipeline was built.

---

## 🚀 Future Roadmap

- [ ] Sentence-transformer embeddings on `overview` for semantic similarity
- [ ] Weighted feature concatenation (director > genre > cast > keywords > overview)
- [ ] Precision@k evaluation using known film clusters (sequels, same-director sets)
- [ ] FAISS approximate nearest neighbor for catalog scaling beyond 10,000 titles
- [ ] Hybrid model: content-based for cold start, collaborative filtering once user history exists
- [ ] Genre and mood filter layer on top of similarity scores
- [ ] Deploy to Streamlit Cloud with persistent similarity matrix (pickle / parquet)

---

## 🤝 Let's Connect

If you are working on recommendation systems, NLP pipelines, or product discovery problems, I would enjoy the conversation.

Feedback on the feature engineering choices or the NLP pipeline is especially welcome. Connect on [LinkedIn](#) or open an issue on this repo.

---

*Built with Python, scikit-learn, NLTK, Streamlit, and the TMDb API*


