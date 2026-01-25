from flask import Flask, make_response, jsonify, request
import pandas as pd

app = Flask(__name__)

@app.route("/upload", methods=["POST"])
def upload_csv():
    if "file" not in request.files:
        return jsonify({"error": "Nenhum arquivo enviado"}), 400

    file = request.files["file"]

    if not file.filename.endswith(".csv"):
        return jsonify({"error": "Arquivo não é CSV"}), 400

    df = pd.read_csv(file)

    return jsonify({
        "message": "CSV recebido com sucesso",
    }), 200

app.run()