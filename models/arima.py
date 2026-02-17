import pandas as pd
from .pre_processing import tratamento_base
from pmdarima import auto_arima
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error
import numpy as np

class ArimaModel(tratamento_base):
    def __init__(self):
         self.df = None
         self.freq = None
         self.treino = None
         self.teste = None
         self.auto_model = None
    
    def avaliar(self, df, treino, teste):
        self.df = df
        self.freq = self.frequencia(df)

        self.treino = treino
        self.teste = teste

        self.treino.set_index('Data', inplace=True)
        self.teste.set_index('Data', inplace=True)

        auto_model = auto_arima(df,
                        start_p=0, start_q=0,
                        max_p=5, max_q=5,
                        m=12,
                        seasonal=True,
                        d=None,
                        trace=True,
                        error_action='ignore',
                        suppress_warnings=True,
                        stepwise=True)
        
        self.auto_model = auto_model

        model = ARIMA(self.treino, order=self.auto_model.order)
        model_fit = model.fit()

        forecast_test = model_fit.forecast(len(self.teste))
        self.df['forecast_manual'] = [None]*len(self.treino)+list(forecast_test)


        # mae = mean_absolute_error(self.teste, forecast_test)
        # mape = mean_absolute_percentage_error(self.teste, forecast_test)
        # rmse = np.sqrt(mean_squared_error(self.teste, forecast_test))

        # print(f'mae - manual: {mae}')
        # print(f'mape - manual: {mape}')
        # print(f'rmse - manual: {rmse}')
    
    def prever_futuro(self):
        self.df.index.freq = self.freq
        model = ARIMA(self.df['Valor'], order=self.auto_model.order)
        model_fit = model.fit() 

        match self.freq:
            case 'D': qtde_pred = 30
            case 'B': qtde_pred = 20
            case 'MS': qtde_pred = 12
            case 'W': qtde_pred = 12
            case _: qtde_pred = 10

        forecast = model_fit.forecast(qtde_pred)
        
        df_forecast = forecast.to_frame(name='forecast')
        df_final = pd.concat([self.df, df_forecast], axis=0)
        df_final.reset_index().to_csv('resultado.csv', index=False, sep=';', encoding='utf-8-sig')
        
        # self.df_resultado = df_final
