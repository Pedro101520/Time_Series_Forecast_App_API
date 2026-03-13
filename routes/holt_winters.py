from flask import jsonify, request, Blueprint
from models.pre_processing import tratamento_base
from models.holt_winters import Holt_Winters_Model
from utils.leitura import ler_arquivo

holt_winters_bp = Blueprint("holt_winters", __name__)

@holt_winters_bp.route("/holt_winters", methods=["POST"])
def rota_holt_winters():
    pipeline = tratamento_base()
    holt_winters = Holt_Winters_Model()

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
        df_tratado = pipeline.retorna()

        holt_winters.avaliar(df_tratado[["Data", "Valor_sem_outliers"]])

        metricas = holt_winters.retorna_metricas()
        modelo = "Holt Winters"
        forecast = holt_winters.prever_futuro()

    except ValueError as e:
        return jsonify({"erro": str(e)}), 400

    return jsonify({
        "message": "Modelo treinado com sucesso",
        "Modelo": f"{modelo}",
        "Metricas": f"{metricas}",
        "Forecast": forecast.to_dict(orient="records")
    }), 200
