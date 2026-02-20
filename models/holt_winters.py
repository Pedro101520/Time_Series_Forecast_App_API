from .pre_processing import tratamento_base
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
import numpy as np

class Holt_Winters_Model(tratamento_base):
    def __init__(self):
        self.df = None
        self.freq = None
        self.treino = None
        self.teste = None
        self.results = None
        self.melhor_modelo = None
    
    def treino_teste(self, df):
        df.reset_index(inplace=True)
        df_avaliar = df[["Data", "Valor"]]

        treino, teste = train_test_split(df_avaliar, test_size=0.2, shuffle=False, random_state=42)
        treino.set_index("Data", inplace=True)
        teste.set_index("Data", inplace=True)

        return treino, teste
        

    def avaliar(self, df):
        df = df
        self.df = df
        self.treino, self.teste = self.treino_teste(df)

        print(self.treino)

        print(self.treino)

        configs = [
            {'trend': 'add', 'seasonal': 'add', 'damped': False},
            {'trend': 'add', 'seasonal': 'mul', 'damped': False},
            {'trend': 'mul', 'seasonal': 'add', 'damped': False},
            {'trend': 'mul', 'seasonal': 'mul', 'damped': False},
            {'trend': 'add', 'seasonal': 'add', 'damped': True},
            {'trend': 'add', 'seasonal': 'mul', 'damped': True},
        ]

        self.results = []

        for cfg in configs:
            try:
                model = ExponentialSmoothing(
                self.treino['Valor'],
                trend=cfg['trend'],
                seasonal=cfg['seasonal'],
                damped_trend=cfg['damped'],
                seasonal_periods=7
                ).fit(optimized=True)

                forecast = model.forecast(len(self.teste))
                rmse = np.sqrt(mean_squared_error(self.teste['Valor'], forecast))

                self.results.append({
                'trend': cfg['trend'],
                'seasonal': cfg['seasonal'],
                'damped': cfg['damped'],
                'rmse': rmse,
                'model': model
            })

            except Exception as e:
                print("Erro:", cfg, e)

        y_true = self.teste['Valor'].values
        y_pred = forecast.values

        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

        print("📊 Métricas de desempenho (Teste)")
        print(f"MAE : {mae:.2f}")
        print(f"RMSE: {rmse:.2f}")
        print(f"MAPE: {mape:.2f}%")


        self.melhor_modelo = min(self.results, key=lambda x: x['rmse'])
    

    def prever_futuro(self):
        self.df["Data"] = pd.to_datetime(self.df["Data"])
        self.df = self.df.set_index("Data")
        self.df = self.df.sort_index()

        freq = pd.infer_freq(self.df.index)
        print(freq)

        self.df = self.df.asfreq(freq)

        model = ExponentialSmoothing(
            self.df['Valor'],
            trend=self.melhor_modelo['trend'],
            seasonal=self.melhor_modelo['seasonal'],
            damped_trend=self.melhor_modelo['damped'],
            seasonal_periods=7
        ).fit(optimized=True)

        forecast = model.forecast(30)

        print(forecast)