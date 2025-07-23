import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
import pickle

# 1. Load & sample your data
df = pd.read_csv("dataset.csv").sample(5000, random_state=42)
df['sentiment'] = df['sentiment'].map({'positive':1,'negative':0})

# 2. Split
X_train, X_test, y_train, y_test = train_test_split(
    df['review'], df['sentiment'], test_size=0.2, random_state=42
)

# 3. Vectorize & train
vect = CountVectorizer(stop_words='english')
X_train_vec = vect.fit_transform(X_train)
model = LogisticRegression(max_iter=1000)
model.fit(X_train_vec, y_train)

# 4. Save the pickles
pickle.dump(vect, open("vectorizer.pkl", "wb"))
pickle.dump(model, open("model.pkl", "wb"))

print("✅ vectorizer.pkl and model.pkl generated.")