import streamlit as st
import pickle
import pandas as pd
import requests
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances, manhattan_distances
from sklearn.preprocessing import binarize
from concurrent.futures import ThreadPoolExecutor

st.set_page_config(page_title="Absolute Cinema", layout="wide")

@st.cache_resource
def load_data():
    movies_dict = pickle.load(open('movie_dict.pkl', 'rb'))
    movies_df = pd.DataFrame(movies_dict)
    movies_df = movies_df.reset_index(drop=True)
    vectors = pickle.load(open('tfidf_vectors.pkl', 'rb'))
    return movies_df, vectors

movies, tfidf_vectors = load_data()

# MASUKKAN API KEY DI SINI
API_KEY = "eba485e014425aca04e53e56b2619ca0"

# Optimasi 1: Menyimpan link poster di RAM agar tidak memanggil API berulang untuk film yang sama
@st.cache_data(show_spinner=False)
def fetch_poster(movie_id):
    url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={API_KEY}&language=en-US"
    try:
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            data = response.json()
            if 'poster_path' in data and data['poster_path']:
                # Optimasi 2: Resolusi diturunkan ke w342 agar payload jaringan lebih ringan
                return "https://image.tmdb.org/t/p/w342/" + data['poster_path']
    except:
        pass
    return "https://via.placeholder.com/342x513?text=No+Poster"

def recommend_movies(movie_title, metric_name):
    movie_index = movies[movies['title'] == movie_title].index[0]
    input_vector = tfidf_vectors[movie_index]
    
    if metric_name == "Cosine Similarity":
        similarity = cosine_similarity(input_vector, tfidf_vectors)[0]
        movies_list = sorted(list(enumerate(similarity)), reverse=True, key=lambda x: x[1])[1:11]
    elif metric_name == "Euclidean Distance":
        distance = euclidean_distances(input_vector, tfidf_vectors)[0]
        movies_list = sorted(list(enumerate(distance)), reverse=False, key=lambda x: x[1])[1:11]
    elif metric_name == "Manhattan Distance":
        distance = manhattan_distances(input_vector, tfidf_vectors)[0]
        movies_list = sorted(list(enumerate(distance)), reverse=False, key=lambda x: x[1])[1:11]
    elif metric_name == "Jaccard Similarity":
        tfidf_binary = binarize(tfidf_vectors)
        input_binary = tfidf_binary[movie_index]
        intersection = tfidf_binary.dot(input_binary.T).toarray().flatten()
        sum_input = input_binary.sum()
        sum_vectors = tfidf_binary.sum(axis=1).A1
        union = sum_input + sum_vectors - intersection
        jaccard_sim = np.divide(intersection, union, out=np.zeros_like(intersection, dtype=float), where=union!=0)
        movies_list = sorted(list(enumerate(jaccard_sim)), reverse=True, key=lambda x: x[1])[1:11]
    elif metric_name == "Pearson Correlation":
        input_dense = input_vector.toarray().flatten()
        pearson_corr = []
        for i in range(tfidf_vectors.shape[0]):
            if i == movie_index:
                pearson_corr.append(1.0)
                continue
            row_dense = tfidf_vectors[i].toarray().flatten()
            corr = np.corrcoef(input_dense, row_dense)[0, 1]
            pearson_corr.append(0.0 if np.isnan(corr) else corr)
        movies_list = sorted(list(enumerate(pearson_corr)), reverse=True, key=lambda x: x[1])[1:11]

    recommended_movies = []
    movie_ids = []
    for i in movies_list:
        movie_ids.append(movies.iloc[i[0]].Movie_id)
        recommended_movies.append(movies.iloc[i[0]].title)
        
    # Optimasi 3: Membuka 10 koneksi API secara serentak (Multi-threading)
    with ThreadPoolExecutor() as executor:
        recommended_posters = list(executor.map(fetch_poster, movie_ids))
        
    return recommended_movies, recommended_posters

if 'selected_movie' not in st.session_state:
    st.session_state.selected_movie = None
if 'selected_movie_id' not in st.session_state:
    st.session_state.selected_movie_id = None
if 'random_movies' not in st.session_state:
    st.session_state.random_movies = movies.sample(10)

st.title('🎬 Absolute Cinema: Recommendation System')

st.sidebar.header("⚙️ Model Configuration")
metric = st.sidebar.selectbox(
    "Choose similarity metrics:",
    ["Cosine Similarity", "Euclidean Distance", "Manhattan Distance", "Jaccard Similarity", "Pearson Correlation"]
)

st.write("### 🔍 Search Movie Reference")
search_query = st.text_input("Type movie title to find reference:", "")

if search_query:
    matched_movies = movies[movies['title'].str.contains(search_query, case=False, na=False)].head(10)
    if not matched_movies.empty:
        st.write("*Search Result:*")
        for row_idx in range(0, len(matched_movies), 5):
            cols = st.columns(5)
            chunk = matched_movies.iloc[row_idx:row_idx+5]
            for col_idx, (_, row) in enumerate(chunk.iterrows()):
                with cols[col_idx]:
                    st.image(fetch_poster(row['Movie_id']), use_container_width=True)
                    display_title = row['title'] if len(row['title']) <= 25 else row['title'][:22] + "..."
                    st.write(f"**{display_title}**")
                    if st.button("Choose", key=f"select_{row['Movie_id']}"):
                        st.session_state.selected_movie = row['title']
                        st.session_state.selected_movie_id = row['Movie_id']
                        st.rerun()
    else:
        st.warning("Movie not found.")

elif not st.session_state.selected_movie:
    st.write("### 🎲 Random Movie")
    for row_idx in range(0, 10, 5):
        cols = st.columns(5)
        chunk = st.session_state.random_movies.iloc[row_idx:row_idx+5]
        for col_idx, (_, row) in enumerate(chunk.iterrows()):
            with cols[col_idx]:
                st.image(fetch_poster(row['Movie_id']), use_container_width=True)
                display_title = row['title'] if len(row['title']) <= 25 else row['title'][:22] + "..."
                st.write(f"**{display_title}**")
                if st.button("Pilih", key=f"rand_{row['Movie_id']}"):
                    st.session_state.selected_movie = row['title']
                    st.session_state.selected_movie_id = row['Movie_id']
                    st.rerun()

if st.session_state.selected_movie:
    st.markdown("---")
    st.write(f"### 🎞️ Movie Reference: **{st.session_state.selected_movie}**")
    
    ref_col1, ref_col2 = st.columns([1, 5])
    with ref_col1:
        st.image(fetch_poster(st.session_state.selected_movie_id), width=150)
    with ref_col2:
        st.write(f"Active Metric: `{metric}`")
        if st.button("Change Reference"):
            st.session_state.selected_movie = None
            st.session_state.selected_movie_id = None
            st.rerun()
            
        if st.button("🚀 10 Movie Recommendation"):
            with st.spinner('Loading...'):
                names, posters = recommend_movies(st.session_state.selected_movie, metric)
                
                st.write("#### Top 10 Movie Recommendation:")
                for row_idx in range(2):
                    rec_cols = st.columns(5)
                    for col_idx in range(5):
                        idx = row_idx * 5 + col_idx
                        with rec_cols[col_idx]:
                            st.image(posters[idx], use_container_width=True)
                            st.write(f"**{names[idx]}**")