import streamlit as st
import pickle
import re
from pathlib import Path

import nltk

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer


# --------------------------------
# Page configuration
# --------------------------------

st.set_page_config(
    page_title="Twitter Sentiment AI",
    page_icon="🐦",
    layout="centered"
)


# --------------------------------
# Custom CSS
# --------------------------------

st.markdown(
    """
    <style>

    .main {
        background-color: #0f172a;
    }

    .title {
        text-align: center;
        font-size: 45px;
        font-weight: bold;
    }

    .subtitle {
        text-align: center;
        color: #94a3b8;
        font-size: 18px;
    }

    .positive {
        color: #22c55e;
        font-size: 32px;
        font-weight: bold;
        text-align: center;
    }

    .negative {
        color: #ef4444;
        font-size: 32px;
        font-weight: bold;
        text-align: center;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# --------------------------------
# Load model
# --------------------------------

BASE_DIR = Path(__file__).resolve().parent


@st.cache_resource
def load_model():

    with open(
        BASE_DIR / "sentiment_model.pkl",
        "rb"
    ) as file:

        model = pickle.load(file)

    with open(
        BASE_DIR / "tfidf_vectorizer.pkl",
        "rb"
    ) as file:

        vectorizer = pickle.load(file)

    return model, vectorizer


model, vectorizer = load_model()


# --------------------------------
# NLP preprocessing
# --------------------------------

stemmer = PorterStemmer()

stop_words = set(
    stopwords.words("english")
)


def preprocess(text):

    text = re.sub(
        r"http\S+|www\S+|https\S+",
        "",
        text
    )

    text = re.sub(
        r"@\w+",
        "",
        text
    )

    text = re.sub(
        r"[^a-zA-Z]",
        " ",
        text
    )

    text = text.lower()

    words = text.split()

    words = [
        stemmer.stem(word)
        for word in words
        if word not in stop_words
    ]

    return " ".join(words)


# --------------------------------
# Header
# --------------------------------

st.markdown(
    '<div class="title">🐦 Twitter Sentiment AI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'AI-powered Tweet Sentiment Analyzer'
    '</div>',
    unsafe_allow_html=True
)

st.write("")


# --------------------------------
# Tweet input
# --------------------------------

tweet = st.text_area(
    "✍️ Enter your tweet",
    placeholder="Example: I absolutely love this new phone!",
    height=150
)


# --------------------------------
# Analyze button
# --------------------------------

if st.button(
    "🔮 ANALYZE SENTIMENT",
    width="stretch"
):

    if tweet.strip() == "":

        st.warning(
            "Please enter a tweet first."
        )

    else:

        # Preprocess
        cleaned_tweet = preprocess(tweet)

        # Vectorize
        tweet_vector = vectorizer.transform(
            [cleaned_tweet]
        )

        # Prediction
        prediction = model.predict(
            tweet_vector
        )[0]

        probabilities = model.predict_proba(
            tweet_vector
        )[0]

        confidence = max(probabilities) * 100


        # --------------------------------
        # Result
        # --------------------------------

        st.divider()

        if prediction == 1:

            st.markdown(
                '<div class="positive">'
                '😊 POSITIVE'
                '</div>',
                unsafe_allow_html=True
            )

        else:

            st.markdown(
                '<div class="negative">'
                '😞 NEGATIVE'
                '</div>',
                unsafe_allow_html=True
            )


        st.metric(
            "Confidence",
            f"{confidence:.2f}%"
        )


        # --------------------------------
        # Probability chart
        # --------------------------------

        st.subheader(
            "📊 Sentiment Probability"
        )

        positive_probability = (
            probabilities[1] * 100
        )

        negative_probability = (
            probabilities[0] * 100
        )


        chart_data = {
            "Negative": negative_probability,
            "Positive": positive_probability
        }

        st.bar_chart(chart_data)


        # --------------------------------
        # Tweet information
        # --------------------------------

        st.subheader(
            "📝 Analyzed Tweet"
        )

        st.info(tweet)