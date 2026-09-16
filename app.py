import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

st.set_page_config(page_title="Netflix AI Recommender", page_icon="🎬", layout="wide")

BASE = Path(__file__).resolve().parent
MODEL = joblib.load(BASE / "netflix_ann_recommender.pkl")
SCALER = joblib.load(BASE / "netflix_recommender_scaler.pkl")
RATINGS = joblib.load(BASE / "netflix_ratings.pkl")
MOODS = joblib.load(BASE / "netflix_moods.pkl")
DF = pd.read_csv(BASE / "netflix_titles.csv")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800;900&display=swap');
*{font-family:Inter,sans-serif}
.stApp{background:#080808;color:#fff}
.block-container{max-width:1450px;padding-top:1.5rem}
.hero{padding:50px;border-radius:12px;background:linear-gradient(90deg,#050505 0%,#180708 55%,#080808 100%);border:1px solid #2a2a2a;margin-bottom:25px}
.hero .red{color:#e50914}
.hero h1{font-size:54px;font-weight:900;margin:8px 0}
.hero p{color:#aaa;max-width:720px}
.card{background:#151515;border:1px solid #292929;border-radius:10px;padding:22px;margin-bottom:15px}
.result{background:linear-gradient(135deg,#26080b,#111);border:1px solid #68151c;border-radius:10px;padding:20px;margin:10px 0}
.movie-title{font-size:23px;font-weight:800}
.small{color:#999;font-size:12px}
div[data-baseweb="select"]>div, input{background:#111!important;color:#fff!important}
.stButton>button{background:#e50914!important;color:#fff!important;border:0!important;font-weight:800!important}
</style>
""", unsafe_allow_html=True)

def mood_vector(text):
    text = str(text).lower()
    # Same mapping used during training
    mapping = {
        "Action":["action","adventure","martial arts","sports"],
        "Comedy":["comedies","comedy","stand-up","humor"],
        "Drama":["dramas","drama","independent movies"],
        "Romance":["romantic","romance"],
        "Thriller":["thrillers","thriller","crime","mysteries"],
        "Horror":["horror","supernatural"],
        "Family":["children","kids","family"],
        "Documentary":["documentaries","documentary"],
        "Feel-Good":["comedies","romantic","family","music","musicals"],
        "Adventure":["adventure","action","fantasy","sci-fi"]
    }
    return [float(any(k in text for k in kws)) for kws in mapping.values()]

def candidate_features(row):
    rating = str(row["rating"]) if pd.notna(row["rating"]) else "Unknown"
    rv = np.zeros(len(RATINGS))
    if rating in list(RATINGS):
        rv[list(RATINGS).index(rating)] = 1
    return np.r_[
        float(row["release_year"] or 0),
        len([v for v in str(row["listed_in"] or "").split(",") if v.strip()]),
        len([v for v in str(row["cast"] or "").split(",") if v.strip()]),
        len([v for v in str(row["country"] or "").split(",") if v.strip()]),
        len(str(row["description"] or "")),
        float(bool(str(row["director"] or "").strip())),
        rv, mood_vector(row["listed_in"])
    ]

st.markdown("""
<div class="hero">
<div style="color:#e50914;font-weight:800;letter-spacing:2px">NETFLIX CONTENT INTELLIGENCE</div>
<h1>Find your next <span class="red">favorite.</span></h1>
<p>Choose your mood, rating and preferred release year. The trained Artificial Neural Network ranks Netflix titles and returns the most relevant movie/show names.</p>
</div>
""", unsafe_allow_html=True)

left,right=st.columns([1,1.4])
with left:
    st.markdown("### 🎯 Your preferences")
    mood=st.selectbox("Mood", MOODS)
    rating=st.selectbox("Rating", list(RATINGS))
    year=st.slider("Release Year", 2000, 2021, 2018)
    content=st.selectbox("Content Type", ["Movie","TV Show","Both"])
    topn=st.slider("Number of recommendations",3,15,8)

    if st.button("✨ Recommend Titles", use_container_width=True):
        candidates=DF.copy()
        candidates["release_year"]=pd.to_numeric(candidates["release_year"],errors="coerce").fillna(0)
        if content!="Both":
            candidates=candidates[candidates["type"]==content]
        if rating!="Unknown":
            # Rating is treated as a preference, not a hard filter, to allow ANN ranking.
            pass

        q_rating=np.zeros(len(RATINGS))
        if rating in list(RATINGS): q_rating[list(RATINGS).index(rating)]=1
        q_mood=np.zeros(len(MOODS)); q_mood[MOODS.index(mood)]=1

        rows=[]
        for _,r in candidates.iterrows():
            cf=candidate_features(r)
            x=np.r_[cf, year, q_rating, q_mood]
            rows.append(x)
        if rows:
            xs=SCALER.transform(np.asarray(rows))
            scores=MODEL.predict(xs)
            candidates=candidates.copy()
            candidates["ANN Score"]=scores
            # Small exact-preference boost for transparent user-facing ranking.
            candidates["Year Match"]=np.maximum(0,1-np.abs(candidates["release_year"]-year)/30)
            candidates["Rating Match"]=(candidates["rating"].fillna("Unknown").astype(str)==rating).astype(float)
            candidates["Mood Match"]=[mood_vector(x)[MOODS.index(mood)] for x in candidates["listed_in"]]
            candidates["Final Score"]=(
                0.70*candidates["ANN Score"]+
                0.15*candidates["Rating Match"]+
                0.15*candidates["Year Match"]
            )
            results=candidates.sort_values("Final Score",ascending=False).head(topn)

            with right:
                st.markdown("### 🍿 Recommended for you")
                for _,r in results.iterrows():
                    st.markdown(f"""
                    <div class="result">
                      <div class="movie-title">🎬 {r['title']}</div>
                      <div class="small">{r['type']} • {int(r['release_year']) if r['release_year'] else 'N/A'} • {r['rating'] or 'Unknown'}</div>
                      <div class="small">{r['listed_in'] or 'Genre unavailable'}</div>
                    </div>
                    """,unsafe_allow_html=True)
        else:
            with right: st.warning("No titles found.")

st.markdown("---")
st.markdown("### 🧠 How the ANN works")
c1,c2,c3=st.columns(3)
c1.info("**Mood**\n\nInferred from Netflix genre metadata.")
c2.info("**Rating**\n\nUsed as a user preference and ranking signal.")
c3.info("**Year**\n\nTitles closer to your selected year receive a higher relevance signal.")

st.caption("Educational ANN recommender. Mood is inferred from the dataset's genre field; Netflix does not provide an official mood label in this dataset.")
