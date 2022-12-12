from flask import Flask, render_template, request, jsonify
import models as ml

app = Flask(__name__)


@app.route("/")
def index():
    return render_template('template.html')


@app.route('/training', methods=['POST', 'GET'])
def training():
    if request.method == "POST":
        file = request.files['file_train']

        # read_data
        raw_docs_train, raw_docs_val = ml.read_data(file)

        # preprocessing
        processed_docs_train = ml.preprocessing(raw_docs_train)
        processed_docs_val = ml.preprocessing(raw_docs_val)

        # tokenize
        word_index, word_seq_train = ml.tokenize_training(processed_docs_train)
        word_seq_test = ml.tokenize_testing(processed_docs_val)

        # word embedding
        embedding_matrix, nb_words, embed_dim = ml.word_embedding(word_index)

        # contoh hasil preprocessing-tokenizing
        n = 100
        cth = processed_docs_train['teks'][n]
        cth_lower = processed_docs_train['lower'][n]
        cth_punctual = processed_docs_train['punctual'][n]
        cth_normalize = processed_docs_train['normalize'][n]
        cth_stemmed = processed_docs_train['stemmed'][n]
        cth_tokenized = str(word_seq_train[n])

        output = {
            'cth': cth,
            'cth_lower': cth_lower,
            'cth_punctual': cth_punctual,
            'cth_normalize': cth_normalize,
            'cth_stemmed': cth_stemmed,
            'cth_tokenized': cth_tokenized
        }

    return jsonify(output)


if __name__ == '__main__':
    app.run(debug=True)
