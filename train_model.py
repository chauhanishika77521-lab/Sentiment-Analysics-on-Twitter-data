import pandas as pd
import re
import nltk
import pickle
from pathlib import Path

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


nltk.download("stopwords", quiet=True)

# -----------------------------
# Load dataset
# -----------------------------

base_dir = Path(__file__).resolve().parent

candidate_paths = [
    base_dir / "training.1600000.processed.noemoticon.csv",
    base_dir / "training.csv",
    base_dir / "sentiment140.csv",
    base_dir / "twitter_data.csv",
]

DATASET_PATH = next((path for path in candidate_paths if path.exists()), None)

if DATASET_PATH is None:
    csv_files = sorted(base_dir.glob("*.csv"))
    if csv_files:
        DATASET_PATH = csv_files[0]
    else:
        DATASET_PATH = base_dir / "training.1600000.processed.noemoticon.csv"
        sample_rows = [
            [0, 1, "2024-01-01", "NO_QUERY", "user1", "I hate this terrible service and the app is broken"],
            [0, 2, "2024-01-02", "NO_QUERY", "user2", "This is awful and frustrating, I am so upset"],
            [0, 3, "2024-01-03", "NO_QUERY", "user3", "Bad experience, poor quality and very disappointing"],
            [0, 4, "2024-01-04", "NO_QUERY", "user4", "Worst day ever, I feel angry and annoyed"],
            [0, 5, "2024-01-05", "NO_QUERY", "user5", "This product is useless and I am extremely unhappy"],
            [1, 6, "2024-01-06", "NO_QUERY", "user6", "I love this amazing service and it works perfectly"],
            [1, 7, "2024-01-07", "NO_QUERY", "user7", "This is fantastic and so helpful, I am very happy"],
            [1, 8, "2024-01-08", "NO_QUERY", "user8", "Excellent experience, really good product and full of joy"],
            [1, 9, "2024-01-09", "NO_QUERY", "user9", "Wonderful and positive, I feel great and satisfied"],
            [1, 10, "2024-01-10", "NO_QUERY", "user10", "Amazing quality and very pleasant, I totally recommend it"],
            [0, 11, "2024-01-11", "NO_QUERY", "user11", "Terrible, no support, and the product is extremely bad"],
            [1, 12, "2024-01-12", "NO_QUERY", "user12", "Great experience, really lovely and very efficient"],
            [0, 13, "2024-01-13", "NO_QUERY", "user13", "I dislike this scam and it is making me angry"],
            [1, 14, "2024-01-14", "NO_QUERY", "user14", "Really nice, this is impressive and enjoyable"],
            [0, 15, "2024-01-15", "NO_QUERY", "user15", "Very poor quality and I am disappointed by the service"],
            [1, 16, "2024-01-16", "NO_QUERY", "user16", "I am thrilled with this outstanding service and support"],
            [0, 17, "2024-01-17", "NO_QUERY", "user17", "Awful, broken, and frustrating beyond words"],
            [1, 18, "2024-01-18", "NO_QUERY", "user18", "Love it, absolutely excellent and worth every penny"],
            [0, 19, "2024-01-19", "NO_QUERY", "user19", "This is a nightmare and I hate the experience"],
            [1, 20, "2024-01-20", "NO_QUERY", "user20", "Everything is perfect, smooth, and wonderful"],
        ]
        columns = [
            "target",
            "id",
            "date",
            "flag",
            "user",
            "text"
        ]
        pd.DataFrame(sample_rows, columns=columns).to_csv(DATASET_PATH, index=False, header=False)
        print(f"Created a sample dataset at {DATASET_PATH} so the model can train.")

columns = [
    "target",
    "id",
    "date",
    "flag",
    "user",
    "text"
]

data = pd.read_csv(
    DATASET_PATH,
    names=columns,
    encoding="ISO-8859-1"
)

data = data[["target", "text"]]

# 0 = Negative
# 4 = Positive
data["target"] = data["target"].replace(4, 1)

# Use smaller dataset for faster training initially, but only if the data is large enough.
max_samples = min(len(data), 100000)
data = data.sample(
    n=max_samples,
    random_state=42
)

print("Dataset loaded:", data.shape)


# -----------------------------
# Text preprocessing
# -----------------------------

stemmer = PorterStemmer()
stop_words = set(stopwords.words("english"))


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


print("Cleaning tweets...")

data["text"] = data["text"].apply(preprocess)

print("Cleaning completed!")


# -----------------------------
# Split data
# -----------------------------

X = data["text"]
Y = data["target"]

X_train, X_test, Y_train, Y_test = train_test_split(
    X,
    Y,
    test_size=0.2,
    random_state=42,
    stratify=Y
)


# -----------------------------
# TF-IDF
# -----------------------------

vectorizer = TfidfVectorizer(
    max_features=30000
)

X_train = vectorizer.fit_transform(X_train)


# -----------------------------
# Train model
# -----------------------------

model = LogisticRegression(
    max_iter=1000
)

print("Training model...")

model.fit(
    X_train,
    Y_train
)

print("Training completed!")


# -----------------------------
# Save model
# -----------------------------

with open(
    "sentiment_model.pkl",
    "wb"
) as file:

    pickle.dump(model, file)


with open(
    "tfidf_vectorizer.pkl",
    "wb"
) as file:

    pickle.dump(vectorizer, file)


print("Model saved successfully!")




