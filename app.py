from flask import Flask, make_response, jsonify, request
import pandas as pd
from models.pre_processing import tratamento_base
from models.prophet import ProphetModel

app = Flask(__name__)

# Ajustar para receber o nome das colunas que serão utiizadas
@app.route("/pipeline/predicao", methods=["POST"])
def upload_csv():
    pipeline = tratamento_base()
    prophet = ProphetModel()

    if "file" not in request.files:
        return jsonify({"error": "Nenhum arquivo enviado"}), 400

    file = request.files["file"]

    if not file.filename.endswith(".csv"):
        return jsonify({"error": "Arquivo não é CSV"}), 400
    

    df = pd.read_csv(file)
    try:
        pipeline.carregar_base(df)
        pipeline.validar_serie()
        pipeline.padroniza_nome()
        pipeline.tratamento_nulo()
        pipeline.tratamento_outliers()
        treino, teste = pipeline.treino_teste()
        df_tratado = pipeline.retorna()

        prophet.padroniza_nome(treino, teste)
        prophet.avaliar(df_tratado[["Data", "Valor"]])
        prophet.prever_futuro()
        df_pred_prophet = prophet.retorna()
    except ValueError as e:
        return jsonify({"erro": str(e)}), 400


    return jsonify({
        "message": "CSV tratado com sucesso",
        # "data": serie_tratada.to_dict(orient="records")
    }), 200


# if __name__ == "__main__":
#     app.run()
app.run()