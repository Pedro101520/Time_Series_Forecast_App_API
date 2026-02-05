import pandas as pd

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

    def padroniza_nome(self):
        colunas = self.df.dtypes
        indice = 0
        count = 0
        for i in range(0, len(colunas)):
            df_teste = self.df.iloc[:,i].astype(str)

            converter = pd.to_datetime(df_teste, errors="coerce", format="mixed")
            if converter.notna().mean() > 0.8:
                self.df.rename(columns={colunas[i]: 'Data'}, inplace=True)
                indice = i
                print("A coluna é de data", i)
                break
            else:
                print("A coluna NÃO é de data", i)
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

            self.df.rename(columns={self.df.dtypes[indice]: 'Valor'}, inplace=True)
            print(self.df.dtypes)

            print(self.df.head())
        else:
            raise ValueError("A série temporal não tem uma coluna de valores")
    
    def tratamento_nulo(self):
        self.df = self.df.sort_values('Data')

        porcentagem_nulo_valor = ((self.df["Valor"].isna().sum())/self.df.shape[0]) * 100
        porcentagem_nulo_data = ((self.df["Data"].isna().sum())/self.df.shape[0]) * 100

        if porcentagem_nulo_data <= 20:
            self.df["Data"] = self.df.dropna(subset=["Data"])

            if porcentagem_nulo_valor <= 20:
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

    # def retorna(self):
    #     print(self.df["Date"])

    