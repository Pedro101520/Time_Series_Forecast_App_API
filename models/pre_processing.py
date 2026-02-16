import pandas as pd
from sklearn.model_selection import train_test_split
from abc import ABC, abstractmethod

class tratamento_base():
    def __init__(self):
        self.df = None

    def carregar_base(self, df):
        self.df = df
    
    def validar_serie(self):
        if len(self.df.columns) != 2:
            raise ValueError("Só é aceito séries temporais com duas colunas - (Data e valor)")
        if self.df.empty:
            raise ValueError("Arquivo vazio")
        if self.df.shape[0] > 200000:
            raise ValueError("Não é aceito séries temporais com mais de 200000 ocorrências")

    def padroniza_nome(self):
        colunas = self.df.columns
        indice = 0
        count = 0
        for i in range(0, len(colunas)):
            df_teste = self.df.iloc[:,i].astype(str)

            converter = pd.to_datetime(df_teste, errors="coerce", format="mixed")
            if converter.notna().mean() > 0.8:
                self.df.rename(columns={colunas[i]: 'Data'}, inplace=True)
                indice = i
                break
            else:
                count += 1
        if count > 1:
            raise ValueError("A série temporal não tem coluna de data")


        if indice > 0:
            indice -= 1
        else:
            indice += 1

        try:
            self.df.iloc[:,indice] = self.df.iloc[:,indice].str.replace(r'[^0-9.,]' , '', regex= True)
        except:
            pass

        if not(self.df.iloc[:,indice].isna().all()):
            col = self.df.columns[indice]
            self.df[col] = pd.to_numeric(self.df[col], errors="coerce")

            self.df.rename(columns={colunas[indice]: 'Valor'}, inplace=True)
        else:
            raise ValueError("A série temporal não tem uma coluna de valores")
    
    def tratamento_nulo(self):
        self.df = self.df.sort_values('Data')

        self.df["Data"] = pd.to_datetime(self.df["Data"], errors="coerce", format="mixed")

        porcentagem_nulo_valor = ((self.df["Valor"].isna().sum())/self.df.shape[0]) * 100
        porcentagem_nulo_data = ((self.df["Data"].isna().sum())/self.df.shape[0]) * 100

        if porcentagem_nulo_data <= 20:
            self.df = self.df.dropna(subset=["Data"])
            self.df.set_index('Data', inplace=True)

            freq = pd.infer_freq(self.df.index)

            if freq is None:
                diffs = self.df.index.to_series().diff().median()
                if diffs == pd.Timedelta(days=1):
                    tem_fim_de_semana = self.df.index.dayofweek.isin([5, 6]).any()
                    
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
                    self.df = self.df.resample('D', on='Data').mean()

            self.df = self.df.asfreq(freq)
            self.df = self.df.reset_index()

            if porcentagem_nulo_valor <= 20:
                self.df = self.df.set_index("Data")
                self.df["Valor"] = self.df["Valor"].interpolate(
                    method='time',
                    limit=5,
                    limit_direction='both'
                )

                self.df["Valor"] = self.df["Valor"].ffill().bfill()
            else:
                raise ValueError("A execução não prosseguirá por conta da alta quantidade de valores nulos que sua série possui (Mesmo com tratamentos a qualidade da predição será inferior)")
        else:
            raise ValueError("A execução não prosseguirá por conta da alta quantidade de valores nulos que sua série possui (Mesmo com tratamentos a qualidade da predição será inferior)")
        self.df = self.df.reset_index()

    def tratamento_outliers(self):
        q1 = self.df["Valor"].quantile(0.25)
        q3 = self.df["Valor"].quantile(0.75)
        iqr = q3-q1

        upper_limit = q3 + (1.5 * iqr)
        lower_limit = q1 - (1.5 * iqr)

        self.df.loc[(self.df["Valor"] > upper_limit) | (self.df["Valor"] < lower_limit)]

        new_df = self.df.loc[(self.df["Valor"] < upper_limit) & (self.df["Valor"] > lower_limit)]

        new_df = self.df.copy()
        new_df.loc[(new_df["Valor"] > upper_limit), "Valor"] = upper_limit
        new_df.loc[(new_df["Valor"] < lower_limit), "Valor"] = lower_limit

        self.df["Valor_sem_outliers"] = new_df["Valor"]
    
    def treino_teste(self):
        df_avaliar = self.df[["Data", "Valor"]]
        treino, teste = train_test_split(df_avaliar, test_size=0.2, shuffle=False, random_state=42)

        return treino, teste

    @abstractmethod
    def prever_futuro(self):
        pass

    @abstractmethod
    def avaliar(self):
        pass

    def retorna(self):
        return(self.df)

    