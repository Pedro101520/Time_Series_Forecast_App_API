from flask import jsonify, request, Blueprint
from models.pre_processing import tratamento_base
from models.prophet import ProphetModel
from models.sarima import SarimaModel
from models.holt_winters import Holt_Winters_Model
from utils.leitura import ler_arquivo

pipeline_bp = Blueprint("pipeline", __name__)

@pipeline_bp.route("/pipeline/predicao", methods=["POST"])
def upload_csv():
    pipeline = tratamento_base()
    prophet = ProphetModel()
    sarima = SarimaModel()
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
        treino, teste = pipeline.treino_teste()
        df_tratado = pipeline.retorna()

        prophet.padroniza_nome(treino, teste)
        prophet.avaliar(df_tratado[["Data", "Valor_sem_outliers"]])

        sarima.avaliar(df_tratado[["Data", "Valor_sem_outliers"]], treino, teste)

        holt_winters.avaliar(df_tratado[["Data", "Valor_sem_outliers"]])

        rmse_compara = []
        rmse_compara.append(prophet.retorna_comparacao())
        rmse_compara.append(sarima.retorna_comparacao())
        rmse_compara.append(holt_winters.retorna_comparacao())

        melhor_rmse = prophet.retorna_comparacao()
        for i in rmse_compara:
            if i < melhor_rmse:
                melhor_rmse = i
        
        melhor_modelo = rmse_compara.index(min(rmse_compara))

        modelo = ""
        metricas = None
        forecast = None
        match melhor_modelo:
            case 0:
                metricas = prophet.retorna_metricas()
                modelo = "Prophet"
                forecast = prophet.prever_futuro()
            case 1:
                metricas = sarima.retorna_metricas()
                modelo = "SARIMA"
                forecast = sarima.prever_futuro()
            case 2:
                metricas = holt_winters.retorna_metricas()
                modelo = "Holt-Winters"
                forecast = holt_winters.prever_futuro()

    except ValueError as e:
        return jsonify({"erro": str(e)}), 400


    return jsonify({
        "message": "Modelo treinado com sucesso",
        "Melhor Modelo": f"{modelo}",
        "Metricas": f"{metricas}",
        "Forecast": forecast.to_dict(orient="records")
    }), 200