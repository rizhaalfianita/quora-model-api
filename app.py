from flask import Flask, jsonify
import pickle
import pandas as pd
from preprocessing import preprocessing_text_with_stemming
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route('/dataset', methods=['GET'])
def get_dataset():
    fix_file = "dataset/fix_data.csv"
    fix_df = pd.read_csv(fix_file)
    fix_data_json = fix_df.to_dict(orient='records')
    return jsonify(fix_data_json), 200

@app.route('/predict/<text>', methods=['GET'])
def predict(text):
    f = open('tfidf_vectorizer.pickle', 'rb')
    tfidf = pickle.load(f)
    f.close()

    w = open('mnb_classifier.pickle', 'rb')
    mnb = pickle.load(w)
    w.close()

    preprocessed_text = preprocessing_text_with_stemming(text)
    transformed_input = tfidf.transform([' '.join(preprocessed_text)])
    predicted_class = mnb.predict(transformed_input.toarray())[0]
    label = "Sincere" if predicted_class == 0 else "Insincere"

    result = {
        "input_text": text,
        "predicted_class": label,
        "tfidf": f"{transformed_input}",
        "preprocessed_text": preprocessed_text
    }

    return jsonify(result), 200

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)