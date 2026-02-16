from .pre_processing import tratamento_base
from prophet import Prophet
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error

class ProphetModel(tratamento_base):
    def __init__(self):
        self.rmse = None
        self.parametros = None
        self.data = None
        self.valor = None
        self.freq = None
        self.df = None
        self.pred = None

    def padroniza_nome(self, treino, teste):
        self.treino = treino.rename(columns={"Data": "ds", "Valor": "y"})
        self.teste  = teste.rename(columns={"Data": "ds", "Valor": "y"})

    def avaliar(self, df):
        self.df = df.rename(columns={"Data": "ds", "Valor": "y"})

        n_test = len(self.teste)

        cps_values = [0.01, 0.05, 0.1, 0.5, 1]
        sps_values = [1, 5, 10, 20]

        melhor_rmse = float('inf')

        df.set_index('Data', inplace=True)
        freq = pd.infer_freq(df.index)

        if freq is None:
            diffs = df.index.to_series().diff().median()
            if diffs == pd.Timedelta(days=1):
                tem_fim_de_semana = df.index.dayofweek.isin([5, 6]).any()
                
                if tem_fim_de_semana:
                    freq = 'D'
                else:
                    freq = 'B'
                    
            elif pd.Timedelta(days=27) <= diffs <= pd.Timedelta(days=31):
                freq = 'MS'
            elif diffs == pd.Timedelta(days=7):
                freq = 'W'
            else:
                freq = 'D'

        self.freq = freq
        df = df.reset_index()

        for cps in cps_values:
            for sps in sps_values:
                modelo = Prophet(
                    changepoint_prior_scale=cps,
                    seasonality_prior_scale=sps,
                    seasonality_mode='multiplicative'
                )

                modelo.fit(self.treino)

                future = modelo.make_future_dataframe(periods=n_test, freq=self.freq)
                forecast = modelo.predict(future)

                y_pred = forecast.tail(n_test)["yhat"]
                rmse = np.sqrt(mean_squared_error(self.teste["y"], y_pred))

                if rmse < melhor_rmse:
                    self.rmse = rmse
                    self.parametros = (cps, sps)

        print("Melhor RMSE:", self.rmse)
        print("Melhores parâmetros:", self.parametros)
    
    def prever_futuro(self):

        modelo = Prophet(
            changepoint_prior_scale=self.parametros[0],
            seasonality_prior_scale=self.parametros[1],
            seasonality_mode='multiplicative'
        )
        
        modelo.fit(self.df)

        qtde_pred = 0
        match self.freq:
            case 'D':
                qtde_pred = 30
            case 'B':
                qtde_pred = 20
            case 'MS':
                qtde_pred = 12
            case 'W':
                qtde_pred = 12

        future = modelo.make_future_dataframe(periods=qtde_pred, freq=self.freq)
        self.pred = forecast = modelo.predict(future)
    
    def retorno_pred(self):
        return self.pred[["ds", "yhat", "yhat_lower", "yhat_upper"]]