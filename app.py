from MeCab_tokenizer import MeCab_tokenizer
from Sentencepiece_tokenizer import Sentencepiece_tokenizer
from Word2Vec_vectorizer import Word2Vec_vectorizer
from GPT_embedding import GPT_embedding
from Cosine_similarity import Cosine_similarity
from DB_connect import DB_connect
from File_processing import File_processing
from Encode_Decode import Encode_Decode
from kss import split_sentences
from flask import Flask, request, jsonify
from flask_cors import CORS
from multiprocessing import Process, Manager
import requests
import uuid


app = Flask(__name__)
CORS(app)

BASE_URL = "http://211.62.99.58:8020"
PRE_PATH = "data"
VOCAB_SIZE = "16000"

mecab = MeCab_tokenizer()
sentencepiece = Sentencepiece_tokenizer()
word2vec = Word2Vec_vectorizer()
gpt = GPT_embedding()
cosine = Cosine_similarity()
db = DB_connect()
file = File_processing(f"{PRE_PATH}/mecab.txt")
ed = Encode_Decode()


def train_model(hyperparameter, movie_data):
    file.remove()
    for plot in movie_data:
        sentences = split_sentences(plot)
        for sentence in sentences:
            mecab_tokens = mecab.tokenize(sentence)
            for mecab_token in mecab_tokens:
                file.write(mecab_token)

    sentencepiece.model_train(
        f"{PRE_PATH}/mecab.txt",
        f"{PRE_PATH}/models/sentencepiece/sentencepiece",
        VOCAB_SIZE,
    )
    sentencepiece_model = sentencepiece.model_load(
        f"{PRE_PATH}/models/sentencepiece/sentencepiece.model"
    )
    plot_tokens = []
    for plot in movie_data:
        sentences = split_sentences(plot)
        for sentence in sentences:
            sentencepiece_tokens = sentencepiece.tokenize(sentencepiece_model, sentence)
            plot_tokens.append(sentencepiece_tokens)

    model_name = str(uuid.uuid4())
    word2vec.model_train_save(
        plot_tokens,
        *(list(hyperparameter.values())[3:]),
        f"{PRE_PATH}/models/word2vec/word2vec-{model_name}.model",
    )

    hyperparameter["name"] = model_name
    requests.post(
        f"{BASE_URL}/api/models/train-complete",
        json=hyperparameter,
    )


@app.route("/train", methods=["POST"])
def trigger_training():
    request_data = request.get_json()

    hyperparameter = request_data.get("parameter")
    movie_data = [movie.get("plot") for movie in request_data.get("movie")]

    p = Process(target=train_model, args=(hyperparameter, movie_data))
    p.start()

    return jsonify({"message": "Request successful"}), 200


def deploy_model(model_id, model_name, table_name, movie_data):
    sentencepiece_model = sentencepiece.model_load(
        f"{PRE_PATH}/models/sentencepiece/sentencepiece.model"
    )
    word2vec_model = word2vec.model_load(
        f"{PRE_PATH}/models/word2vec/word2vec-{model_name}.model"
    )

    plots_token = []
    for movie_id, plot in movie_data:
        plot_token = []
        sentences = split_sentences(plot)
        for sentence in sentences:
            sentencepiece_tokens = sentencepiece.tokenize(sentencepiece_model, sentence)
            plot_token.append(sentencepiece_tokens)
        plots_token.append((movie_id, plot_token))

    db.truncate(table_name)
    for movie_id, plot_token in plots_token:
        word2vec_vector_list = word2vec.vectorize(word2vec_model, plot_token)
        for word2vec_vector in word2vec_vector_list:
            word2vec_vector_string = ed.encode(word2vec_vector)
            db.insert(
                f"INSERT INTO {table_name} (movie_id, vector) VALUES (%s, %s)",
                (
                    movie_id,
                    word2vec_vector_string,
                ),
            )

    requests.post(f"{BASE_URL}/api/models/deploy-complete?id={model_id}")


@app.route("/<int:model_id>/deploy", methods=["POST"])
def trigger_deploy(model_id):
    request_data = request.get_json()

    model_name = request_data.get("modelName")
    table_name = request_data.get("tableName")
    movie_data = [
        (movie.get("id"), movie.get("plot")) for movie in request_data.get("movie")
    ]

    p = Process(
        target=deploy_model, args=(model_id, model_name, table_name, movie_data)
    )
    p.start()

    return jsonify({"message": "Request successful"}), 200


def resultWord2Vec(user_input, str_embedding_data, model_name, result_list):
    sentencepiece_model = sentencepiece.model_load(
        f"{PRE_PATH}/models/sentencepiece/sentencepiece.model"
    )
    word2vec_model = word2vec.model_load(
        f"{PRE_PATH}/models/word2vec/word2vec-{model_name}.model"
    )

    user_input_token = sentencepiece.tokenize(sentencepiece_model, user_input)
    user_vector_list = word2vec.vectorize(word2vec_model, [user_input_token])[0]
    embedding_data = [
        (str_vector.get("movieId"), ed.decode(str_vector.get("vector")))
        for str_vector in str_embedding_data
    ]

    movie_list = cosine.find_most_similar_movies(user_vector_list, embedding_data, 10)
    for movie_id, similarity in movie_list:
        result_list.append((movie_id, similarity))

def resultGpt(user_input, str_embedding_data, result_list):
    user_input_vector = gpt.vectorize(user_input)
    embedding_data = [
        (str_vector.get("movieId"), ed.decode(str_vector.get("vector")))
        for str_vector in str_embedding_data
    ]

    movie_list = cosine.find_most_similar_movies(user_input_vector, embedding_data, 10)
    for movie_id, similarity in movie_list:
        result_list.append((movie_id, similarity))


@app.route("/result", methods=["POST"])
def trigger_result():
    request_data = request.get_json()

    manager = Manager()
    result_list = manager.list()

    user_input = request_data["input"]
    str_embedding_data = request_data["embeddingVector"]
    if "modelName" in dict(request_data).keys():
        model_name = request_data["modelName"]
        p = Process(
            target=resultWord2Vec, args=(user_input, str_embedding_data, model_name, result_list)
        )
    else:
        p = Process(
            target=resultGpt, args=(user_input, str_embedding_data, result_list)
        )
    p.start()
    p.join()

    return (
        jsonify(
            {
                "input": user_input,
                "output": [
                    {
                        "movieId": movie_id,
                        "similarity": similarity,
                    }
                    for movie_id, similarity in result_list
                ],
            },
        ),
        200,
    )


if __name__ == "__main__":
    app.run()
