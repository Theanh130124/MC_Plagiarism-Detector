from flask import Flask, render_template, request, redirect, url_for, abort, flash, send_from_directory
import pickle
app = Flask(__name__)

model = pickle.load(open('model.pkl', 'rb'))
tfidf_vectorizer = pickle.load(open('vector.pkl', 'rb'))

def detect(input_text):
    vectorized_text = tfidf_vectorizer.transform([input_text])
    result = model.predict(vectorized_text)
    return "Plagiarism Detected" if result[0] == 1 else "No Plagiarism Detected"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect_plagiarism():
    input_text = request.form['text']
    detection_result = detect(input_text)
    return render_template('index.html', result=detection_result)

if __name__=='__main__':
    app.run(debug=True)