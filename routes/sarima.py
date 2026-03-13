from flask import jsonify, request, Blueprint
from models.pre_processing import tratamento_base
from models.sarima import SarimaModel
from utils.leitura import ler_arquivo

sarima_bp = Blueprint("sarima", __name__)

@sarima_bp.route("/sarima", methods=["POST"])
def rota_sarima():
    pipeline = tratamento_base()
    sarima = SarimaModel()

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

        sarima.avaliar(df_tratado[["Data", "Valor_sem_outliers"]], treino, teste)

        metricas = sarima.retorna_metricas()
        modelo = "Sarima"
        forecast = sarima.prever_futuro()

    except ValueError as e:
        return jsonify({"erro": str(e)}), 400

    return jsonify({
        "message": "Modelo treinado com sucesso",
        "Modelo": f"{modelo}",
        "Metricas": f"{metricas}",
        "Forecast": forecast.to_dict(orient="records")
    }), 200
