import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
data = {
    'news': [
        'Government launches new education policy',
        'Aliens landed in India yesterday',
        'Stock market reaches all time high',
        'Drinking hot water cures all diseases',
        'Scientists discover new vaccine',
        'Moon will turn pink tomorrow night'
    ],
    'label': [1, 0, 1, 0, 1, 0]
}
df = pd.DataFrame(data)
X = df['news']
y = df['label']
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
news = input("Enter News: ")
news_vector = vectorizer.transform([news])
prediction = model.predict(news_vector)
if prediction[0] == 1:
    print("Real News")
else:
    print("Fake News")