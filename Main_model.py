import pickle as pk
import re
import nltk
nltk.download("stopwords")
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from flask import Flask, request, jsonify
from flask_cors import CORS
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "Logistic_Regression.pk")
vectorizer_path = os.path.join(BASE_DIR, "CountVectorizer.pk")

with open(model_path, "rb") as file:
    model = pk.load(file)

with open(vectorizer_path, "rb") as file:
    cv = pk.load(file)

ps = PorterStemmer()
stop_words = set(stopwords.words("english"))

def clean_url(link):
    review = re.sub("[^a-zA-Z]", " ", link)
    review = review.lower().split()
    review = [ps.stem(word) for word in review if word not in stop_words]
    return " ".join(review)

def predict(link):
    cleaned = clean_url(link)
    vectorized = cv.transform([cleaned]).toarray()
    predicted_outcome = model.predict(vectorized)
    return predicted_outcome[0]

app = Flask(__name__)
CORS(app)

@app.route("/predict", methods=["POST"])
def predict_route():
    input_data = request.get_json()
    link = str(input_data["link"])
    result = predict(link)
    return jsonify({"Predicted_outcome": int(result)})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)