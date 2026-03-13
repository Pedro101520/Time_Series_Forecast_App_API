from flask import jsonify, request, Blueprint
from models.pre_processing import tratamento_base
from models.prophet import ProphetModel
from utils.leitura import ler_arquivo

prophet_bp = Blueprint("prophet", __name__)

@prophet_bp.route("/prophet", methods=["POST"])
def rota_prophet():
    pipeline = tratamento_base()
    prophet = ProphetModel()

    if "file" not in request.files:
        return jsonify({"error": "Nenhum arquivo enviado"}), 400

    file = request.files["file"]

    if not file.filename.endswith(".csv"):
        return jsonify({"error": "Arquivo não é CSV"}), 400
    
    df = ler_arquivo(file)
    
    try:
        pipeline.carregar_base(df)
        pipeline.validar_serie()
        pipeline.padroniza_nome()
        pipeline.tratamento_nulo()
        pipeline.tratamento_outliers()
        treino, teste = pipeline.treino_teste()
        df_tratado = pipeline.retorna()

        prophet.padroniza_nome(treino, teste)
        prophet.avaliar(df_tratado[["Data", "Valor_sem_outliers"]])

        metricas = prophet.retorna_metricas()
        modelo = "Prophet"
        forecast = prophet.prever_futuro()

    except ValueError as e:
        return jsonify({"erro": str(e)}), 400

    return jsonify({
        "message": "Modelo treinado com sucesso",
        "Modelo": f"{modelo}",
        "Metricas": f"{metricas}",
        "Forecast": forecast.to_dict(orient="records")
    }), 200
