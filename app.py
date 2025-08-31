from flask import Flask, render_template, request
import joblib

# Load model and vectorizer
model = joblib.load("fake_news_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")   # webpage

@app.route("/predict", methods=["POST"])
def predict():
    if request.method == "POST":
        news_text = request.form["news"]   # Get input from form
        transformed = vectorizer.transform([news_text])
        prediction = model.predict(transformed)[0]

        result = "Real News ✅" if prediction == 1 else "Fake News ❌"
        return render_template("index.html", news=news_text, result=result)

if __name__ == "__main__":
    app.run(debug=True)
