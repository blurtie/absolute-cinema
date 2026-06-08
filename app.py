import streamlit as st
import pickle
import pandas as pd
import requests
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances, manhattan_distances
from concurrent.futures import ThreadPoolExecutor

# ══════════════════════════════════════════════════════════════════════
#  PAGE CONFIG
# ══════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Absolute Cinema",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ══════════════════════════════════════════════════════════════════════
#  GLOBAL STYLES
# ══════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@600;700&family=Outfit:wght@300;400;500&display=swap');

/* ── Base ─────────────────────────────────────────────────────── */
.stApp, [data-testid="stAppViewContainer"] {
    background: #080808;
    color: #e0e0e0;
    font-family: 'Outfit', sans-serif;
}
[data-testid="stHeader"] { background: transparent; }
#MainMenu, footer, header { visibility: hidden; }

/* ── Brand ────────────────────────────────────────────────────── */
.brand-eyebrow {
    font-size: 0.7rem;
    letter-spacing: 6px;
    text-transform: uppercase;
    color: #3d3d3d;
    margin-bottom: 6px;
}
.brand-title {
    font-family: 'Cormorant Garamond', serif;
    font-size: 4rem;
    font-weight: 700;
    color: #f5f5f5;
    line-height: 1;
    margin: 0;
}
.brand-gold { color: #d40019; }
.brand-tagline {
    color: #3d3d3d;
    font-size: 0.82rem;
    font-weight: 300;
    margin-top: 10px;
    letter-spacing: 1px;
}

/* ── Section header ───────────────────────────────────────────── */
.sec-hdr {
    font-family: 'Cormorant Garamond', serif;
    font-size: 1.45rem;
    color: #f0f0f0;
    border-left: 3px solid #d40019;
    padding-left: 12px;
    margin: 8px 0 16px 0;
}

/* ── Movie label ──────────────────────────────────────────────── */
.mv-label {
    font-size: 0.71rem;
    color: #888;
    text-align: center;
    padding: 5px 4px 2px 4px;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
}

/* ── Selected film panel ──────────────────────────────────────── */
.sel-panel {
    background: linear-gradient(135deg, #141414, #1a1a0d);
    border: 1px solid rgba(201,168,76,.2);
    border-radius: 12px;
    padding: 18px 22px;
}
.sel-eyebrow {
    font-size: 0.68rem;
    letter-spacing: 4px;
    text-transform: uppercase;
    color: #d40019;
    margin-bottom: 6px;
}
.sel-title {
    font-family: 'Cormorant Garamond', serif;
    font-size: 2.2rem;
    color: #fff;
    margin: 0;
    line-height: 1.15;
}

/* ── Score chip ───────────────────────────────────────────────── */
.chip {
    background: rgba(201,168,76,.1);
    border: 1px solid rgba(201,168,76,.28);
    color: #c9a84c;
    border-radius: 50px;
    padding: 2px 9px;
    font-size: 0.67rem;
    display: inline-block;
    margin-top: 3px;
}

/* ── Metric info strip ────────────────────────────────────────── */
.metric-info {
    background: #111;
    border-left: 2px solid rgba(201,168,76,.35);
    border-radius: 0 6px 6px 0;
    padding: 10px 14px;
    font-size: 0.81rem;
    color: #666;
    margin-bottom: 16px;
}
.metric-info strong { color: #c9a84c; }

/* ── Comparison table ─────────────────────────────────────────── */
.cmp-table { width: 100%; border-collapse: collapse; font-size: 0.79rem; }
.cmp-table th {
    background: #111;
    color: #c9a84c;
    padding: 9px 11px;
    text-align: left;
    font-weight: 500;
    font-size: 0.67rem;
    letter-spacing: 2px;
    text-transform: uppercase;
    border-bottom: 1px solid #222;
}
.cmp-table td {
    padding: 9px 11px;
    border-bottom: 1px solid #181818;
    color: #bbb;
    vertical-align: top;
    line-height: 1.5;
}

/* ── Buttons ──────────────────────────────────────────────────── */
.stButton > button {
    background: #c9a84c !important;
    color: #080808 !important;
    border: none !important;
    border-radius: 6px !important;
    font-family: 'Outfit', sans-serif !important;
    font-weight: 500 !important;
    transition: background .2s, transform .1s !important;
}
.stButton > button:hover {
    background: #d40019 !important;
    transform: translateY(-1px) !important;
}
.stButton > button:disabled {
    background: rgba(212, 0, 25, .2) !important;
    color: #d40019 !important;
}

/* ── Text input ───────────────────────────────────────────────── */
.stTextInput input {
    background: #111 !important;
    color: #f0f0f0 !important;
    border-color: #242424 !important;
    border-radius: 8px !important;
    font-family: 'Outfit', sans-serif !important;
    font-size: 0.9rem !important;
}
.stTextInput input:focus {
    border-color: #c9a84c !important;
    box-shadow: 0 0 0 2px rgba(201,168,76,.12) !important;
}

/* ── Selectbox ────────────────────────────────────────────────── */
.stSelectbox [data-baseweb="select"] > div {
    background: #111 !important;
    border-color: #242424 !important;
    border-radius: 8px !important;
    color: #f0f0f0 !important;
}

/* ── Divider ──────────────────────────────────────────────────── */
[data-testid="stDivider"] hr { border-color: #181818; }

/* ── Caption ──────────────────────────────────────────────────── */
.stCaption p { color: #484848 !important; font-size: 0.75rem !important; }

/* ── Tabs ─────────────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
    background: transparent;
    border-bottom: 1px solid #1e1e1e;
    gap: 2px;
}
.stTabs [data-baseweb="tab"] {
    color: #484848;
    font-family: 'Outfit', sans-serif;
    font-size: 0.8rem;
    padding: 6px 14px;
    border-radius: 6px 6px 0 0;
}
.stTabs [data-baseweb="tab"][aria-selected="true"] {
    color: #c9a84c;
    background: rgba(201,168,76,.06);
    border-bottom: 2px solid #c9a84c;
}

/* ── Spinner ──────────────────────────────────────────────────── */
.stSpinner > div { border-top-color: #c9a84c !important; }

/* ── Info box ─────────────────────────────────────────────────── */
.stAlert { background: #111 !important; border-color: #242424 !important; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
#  DATA LOADING  (cached — loaded once per session)
# ══════════════════════════════════════════════════════════════════
@st.cache_resource
def load_data():
    movies_dict = pickle.load(open('movie_dict.pkl', 'rb'))
    movies      = pd.DataFrame(movies_dict)
    vectors     = pickle.load(open('tfidf_vectors.pkl', 'rb'))
    return movies, vectors


@st.cache_resource
def get_dense_matrix():
    """
    Converts the sparse TF-IDF matrix (~400 MB dense) once and cached it.
    Used only by rec_pearson — first call is slow, subsequent calls are instant.
    """
    _, vecs = load_data()
    return vecs.toarray()


movies, tfidf_vectors = load_data()


# ══════════════════════════════════════════════════════════════════
#  SESSION STATE DEFAULTS
# ══════════════════════════════════════════════════════════════════
# State untuk menyimpan 10 film acak agar tidak re-render saat tombol diklik
if 'random_movies' not in st.session_state:
    st.session_state.random_movies = movies.sample(10)

_DEFAULTS = {
    'selected_movie':  None,
    'show_recs':       False,
    'comparison_done': False,
    'comparison_data': {},
}
for _k, _v in _DEFAULTS.items():
    st.session_state.setdefault(_k, _v)


# ══════════════════════════════════════════════════════════════════
#  TMDB API  (cached 24h)
# ══════════════════════════════════════════════════════════════════
_API_KEY = "eba485e014425aca04e53e56b2619ca0"

@st.cache_data(ttl=86400)
def fetch_poster(movie_id: int) -> str:
    try:
        url  = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={_API_KEY}&language=en-US"
        data = requests.get(url, timeout=5).json()
        path = data.get('poster_path')
        if path:
            return f"https://image.tmdb.org/t/p/w342{path}"
    except Exception:
        pass
    return "https://placehold.co/342x513/141414/c9a84c?text=No+Poster"


def fetch_posters_parallel(movie_ids: list) -> dict:
    """Fungsi pembantu untuk mengunduh banyak poster secara serentak (Multi-threading)."""
    unique_ids = list(set(movie_ids))
    with ThreadPoolExecutor(max_workers=10) as executor:
        urls = list(executor.map(fetch_poster, unique_ids))
    return dict(zip(unique_ids, urls))


# ══════════════════════════════════════════════════════════════════
#  RECOMMENDATION ENGINES
# ══════════════════════════════════════════════════════════════════
def _build_top(scores: np.ndarray, descending: bool, skip_idx: int, n: int):
    """Sort scores, skip the query film itself, return top-n tuples."""
    ranked = sorted(enumerate(scores),
                    key=lambda x: -x[1] if descending else x[1])
    return [
        (movies.iloc[i]['title'], int(movies.iloc[i]['Movie_id']), float(s))
        for i, s in ranked
        if i != skip_idx
    ][:n]


def rec_cosine(idx: int, n: int = 10):
    sim = cosine_similarity(tfidf_vectors[idx], tfidf_vectors)[0]
    return _build_top(sim, True, idx, n)


def rec_euclidean(idx: int, n: int = 10):
    dist = euclidean_distances(tfidf_vectors[idx], tfidf_vectors)[0]
    return _build_top(dist, False, idx, n)


def rec_jaccard(idx: int, n: int = 10):
    iv    = (tfidf_vectors[idx] > 0).astype(int)
    bv    = (tfidf_vectors > 0).astype(int)
    inter = bv.dot(iv.T).toarray().flatten()
    union = iv.sum() + bv.sum(axis=1).A1 - inter
    jac   = np.where(union > 0, inter / union, 0.0)
    return _build_top(jac, True, idx, n)


def rec_manhattan(idx: int, n: int = 10):
    dist = manhattan_distances(tfidf_vectors[idx], tfidf_vectors)[0]
    return _build_top(dist, False, idx, n)


def rec_pearson(idx: int, n: int = 10):
    dm       = get_dense_matrix()          # ~400 MB, loaded once
    inp      = dm[idx]
    inp_c    = inp - inp.mean()
    inp_norm = np.linalg.norm(inp_c)

    all_c    = dm - dm.mean(axis=1, keepdims=True)
    all_norm = np.linalg.norm(all_c, axis=1)

    num   = all_c @ inp_c
    denom = all_norm * inp_norm
    corr  = np.where(denom > 0, num / denom, 0.0)
    return _build_top(corr, True, idx, n)


# ── Metric registry ────────────────────────────────────────────────
METRICS: dict = {
    "Cosine Similarity": {
        "fn":     rec_cosine,
        "unit":   "Similarity",
        "better": "Higher → More similar",
        "desc":   (
            "Measures the cosine of the angle between TF-IDF vectors. "
            "Magnitude-agnostic — ideal for sparse text representations."
        ),
        "slow": False,
    },
    "Euclidean Distance": {
        "fn":     rec_euclidean,
        "unit":   "Distance",
        "better": "Lower → More similar",
        "desc":   (
            "Straight-line distance in 5,000-dimensional TF-IDF space. "
            "Magnitude-sensitive; longer overviews can create distance bias."
        ),
        "slow": False,
    },
    "Jaccard Similarity": {
        "fn":     rec_jaccard,
        "unit":   "Similarity",
        "better": "Higher → More similar",
        "desc":   (
            "Overlap ratio of binarised vocabulary sets (intersection ÷ union). "
            "Captures pure keyword overlap — ignores term weights."
        ),
        "slow": False,
    },
    "Manhattan Distance": {
        "fn":     rec_manhattan,
        "unit":   "Distance",
        "better": "Lower → More similar",
        "desc":   (
            "City-block (L1) distance between vectors. "
            "More robust to extreme term weights than Euclidean on sparse data."
        ),
        "slow": False,
    },
    "Pearson Correlation": {
        "fn":     rec_pearson,
        "unit":   "Correlation",
        "better": "Higher → More similar",
        "desc":   (
            "Linear correlation of TF-IDF weight profiles. "
            "Requires a dense matrix conversion (~400 MB) — first call may take 10–30 s."
        ),
        "slow": True,
    },
}


# ══════════════════════════════════════════════════════════════════
#  UI HELPERS
# ══════════════════════════════════════════════════════════════════
def render_movie_grid(pairs: list, n_cols: int = 5, max_n: int = 10):
    """
    Renders a poster grid from a list of (title, movie_id) pairs.
    """
    subset = pairs[:max_n]
    
    # Menarik seluruh poster secara paralel sebelum proses rendering visual
    mids = [mid for _, mid in subset]
    poster_dict = fetch_posters_parallel(mids)
    
    for chunk in [subset[i:i + n_cols] for i in range(0, len(subset), n_cols)]:
        cols = st.columns(n_cols)
        for col, (title, mid) in zip(cols, chunk):
            with col:
                st.image(poster_dict[mid], use_container_width=True)
                st.markdown(f'<div class="mv-label">{title}</div>', unsafe_allow_html=True)
                is_sel  = st.session_state.selected_movie == title
                btn_lbl = "✓ Selected" if is_sel else "Select"
                if st.button(btn_lbl, key=f"g_{mid}", use_container_width=True, disabled=is_sel):
                    st.session_state.selected_movie  = title
                    st.session_state.show_recs       = False
                    st.session_state.comparison_done = False
                    st.session_state.comparison_data = {}
                    st.rerun()


def render_rec_row(recs: list, unit: str):
    """Renders 10 recommendation cards (2 rows of 5) with score chips."""
    # Menarik 10 poster secara paralel
    mids = [mid for _, mid, _ in recs]
    poster_dict = fetch_posters_parallel(mids)
    
    for i in range(0, len(recs), 5):
        cols = st.columns(5)
        for col, (title, mid, score) in zip(cols, recs[i:i+5]):
            with col:
                st.image(poster_dict[mid], use_container_width=True)
                st.markdown(f'<div class="mv-label">{title}</div>', unsafe_allow_html=True)
                st.markdown(
                    f'<div style="text-align:center">'
                    f'<span class="chip">{unit}: {score:.4f}</span></div>',
                    unsafe_allow_html=True,
                )


# ══════════════════════════════════════════════════════════════════
#  ── 1. HEADER ────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════
st.markdown("""
<div style="padding:2.5rem 0 1rem 0">
    <p class="brand-eyebrow">🎬 &nbsp; Film Recommendation System</p>
    <h1 class="brand-title">Absolute <span class="brand-gold">Cinema</span></h1>
    <p class="brand-tagline">
        Content-based discovery &nbsp;·&nbsp; TF-IDF vectorisation &nbsp;·&nbsp;
        5 similarity metrics &nbsp;·&nbsp; 9,974 films
    </p>
</div>
""", unsafe_allow_html=True)

st.divider()

# ══════════════════════════════════════════════════════════════════
#  ── 2. SEARCH + METRIC SELECTOR ─────────────────────────────────
# ══════════════════════════════════════════════════════════════════
s_col, m_col = st.columns([3, 1])

with s_col:
    query = st.text_input(
        "search",
        placeholder="🔍  Search a film by title…",
        label_visibility="collapsed",
    )
with m_col:
    chosen = st.selectbox(
        "Metric",
        options=list(METRICS.keys()),
        index=0,
        help="Similarity metric used to rank the recommendations.",
    )

# ══════════════════════════════════════════════════════════════════
#  ── 3. MOVIE BROWSER GRID ────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════
st.markdown('<div class="sec-hdr">Browse Films</div>', unsafe_allow_html=True)

if query.strip():
    hits = movies[movies['title'].str.contains(query.strip(), case=False, na=False)]
    if hits.empty:
        st.info(f"No results for **'{query}'** — try a different title.")
    else:
        n_total = len(hits)
        st.caption(f"Showing {min(n_total, 10)} of {n_total} results")
        render_movie_grid(
            list(zip(hits['title'].values, hits['Movie_id'].values))
        )
else:
    r_col1, r_col2 = st.columns([8, 1])
    with r_col1:
        st.caption("🎲 Random films  ·  Type above to search by title")
    with r_col2:
        if st.button("↻ Shuffle", use_container_width=True):
            st.session_state.random_movies = movies.sample(10)
            st.rerun()
            
    rand_10 = st.session_state.random_movies
    render_movie_grid(
        list(zip(rand_10['title'].values, rand_10['Movie_id'].values))
    )

st.divider()

# ══════════════════════════════════════════════════════════════════
#  ── 4. SELECTED FILM PANEL ───────────────────────────────────────
# ══════════════════════════════════════════════════════════════════
sel = st.session_state.selected_movie

if sel:
    sel_row = movies[movies['title'] == sel].iloc[0]
    sel_id  = int(sel_row['Movie_id'])

    p_col, info_col = st.columns([1, 5])
    with p_col:
        st.image(fetch_poster(sel_id), width=115)
    with info_col:
        st.markdown(f"""
        <div class="sel-panel">
            <p class="sel-eyebrow">Selected Film</p>
            <h2 class="sel-title">{sel}</h2>
        </div>""", unsafe_allow_html=True)

    st.markdown("")

    b1, b2, _ = st.columns([2, 2, 6])
    with b1:
        if st.button("🎬  Get Recommendations", use_container_width=True):
            st.session_state.show_recs       = True
            st.session_state.comparison_done = False
            st.session_state.comparison_data = {}
    with b2:
        if st.button("✕  Clear Selection", use_container_width=True):
            st.session_state.selected_movie  = None
            st.session_state.show_recs       = False
            st.session_state.comparison_done = False
            st.session_state.comparison_data = {}
            st.rerun()

# ══════════════════════════════════════════════════════════════════
#  ── 5. RECOMMENDATIONS ───────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════
if st.session_state.show_recs and sel:
    sel_idx = int(movies[movies['title'] == sel].index[0])
    m       = METRICS[chosen]

    st.divider()
    st.markdown(
        f'<div class="sec-hdr">Recommendations — {chosen}</div>',
        unsafe_allow_html=True,
    )

    st.markdown(f"""
    <div class="metric-info">
        <strong>{m["better"]}</strong> &nbsp;|&nbsp; {m["desc"]}
    </div>""", unsafe_allow_html=True)

    if m["slow"]:
        st.caption(
            "⚠️  First run with Pearson may take 10–30 s while the dense matrix "
            "is built and cached in memory."
        )

    with st.spinner(f"Computing {chosen}…"):
        primary_recs = m["fn"](sel_idx, n=10) # Disesuaikan menjadi 10 film

    render_rec_row(primary_recs, m["unit"])

    # ════════════════════════════════════════════════════════════
    #  ── 6. METRIC COMPARISON ────────────────────────────────────
    # ════════════════════════════════════════════════════════════
    st.divider()

    cmp_col, note_col = st.columns([2, 4])
    with cmp_col:
        run_cmp = st.button("📊  Compare All 5 Metrics", use_container_width=True)
    with note_col:
        st.caption(
            "Runs all 5 similarity metrics on this film and shows ranked results "
            "side-by-side. Pearson may add ~30 s on the first run."
        )

    if run_cmp:
        comp: dict = {}
        for mname, mdata in METRICS.items():
            with st.spinner(f"Running {mname}…"):
                comp[mname] = mdata["fn"](sel_idx, n=5) # Komparasi dibatasi 5 agar UI muat
        st.session_state.comparison_data = comp
        st.session_state.comparison_done = True

    if st.session_state.comparison_done and st.session_state.comparison_data:
        comp = st.session_state.comparison_data

        st.markdown(
            '<div class="sec-hdr" style="font-size:1.2rem;margin-top:20px">'
            'Metric Comparison Results</div>',
            unsafe_allow_html=True,
        )
        st.caption(
            f"Top-5 recommendations for **{sel}** — ranked independently by each metric."
        )

        # Pre-fetch seluruh poster untuk tab komparasi secara serentak
        all_comp_mids = [mid for results in comp.values() for _, mid, _ in results]
        comp_poster_dict = fetch_posters_parallel(all_comp_mids)

        # ── Tabs: one per metric with poster grid ────────────────
        tabs = st.tabs(list(comp.keys()))
        for tab, (mname, results) in zip(tabs, comp.items()):
            with tab:
                md = METRICS[mname]
                st.markdown(
                    f'<div class="metric-info" style="margin-top:12px">'
                    f'<strong>{md["better"]}</strong> &nbsp;|&nbsp; {md["desc"]}'
                    f'</div>',
                    unsafe_allow_html=True,
                )
                t_cols = st.columns(5)
                for t_col, (title, mid, score) in zip(t_cols, results):
                    with t_col:
                        st.image(comp_poster_dict[mid], use_container_width=True)
                        st.markdown(
                            f'<div class="mv-label">{title}</div>',
                            unsafe_allow_html=True,
                        )
                        st.markdown(
                            f'<div style="text-align:center">'
                            f'<span class="chip">{score:.4f}</span></div>',
                            unsafe_allow_html=True,
                        )

        st.divider()

        # ── Side-by-side ranking table ───────────────────────────
        st.markdown("**Side-by-Side Ranking Summary**")

        headers   = "<tr><th>Rank</th>" + \
                    "".join(f"<th>{mname}</th>" for mname in comp) + "</tr>"
        rows_html = ""
        for ri in range(5):
            rank_lbl  = f"#{ri + 1}"
            row       = (f"<tr><td style='color:#c9a84c;font-weight:600'>"
                         f"{rank_lbl}</td>")
            for mname, results in comp.items():
                if ri < len(results):
                    title, _, score = results[ri]
                    unit = METRICS[mname]["unit"]
                else:
                    title, score, unit = "—", 0.0, ""
                row += (f"<td>{title}<br>"
                        f"<span class='chip'>{score:.3f}</span></td>")
            row += "</tr>"
            rows_html += row

        st.markdown(
            f'<div style="overflow-x:auto;margin-top:12px">'
            f'<table class="cmp-table">'
            f'<thead>{headers}</thead>'
            f'<tbody>{rows_html}</tbody>'
            f'</table></div>',
            unsafe_allow_html=True,
        )