# tratamento_dados.py
import pandas as pd


def carregar_e_tratar_dados(caminho_csv="Teen_Mental_Health_Dataset.csv"):
    df = pd.read_csv(caminho_csv, sep=";")

    # 1. Dicionário para as colunas reais da tabela
    traducao_colunas = {
        "age": "Idade",
        "gender": "Genero",
        "daily_social_media_hours": "Horas_Redes_Sociais_Diarias",
        "platform_usage": "Plataforma_Mais_Usada",
        "sleep_hours": "Horas_de_Sono",
        "screen_time_before_sleep": "Tempo_de_Tela_Antes_De_Dormir",
        "academic_performance": "Desempenho_Academico",
        "physical_activity": "Atividade_Fisica_Horas",
        "social_interaction_level": "Nivel_Interacao_Social",
        "stress_level": "Nivel_de_Estresse",
        "anxiety_level": "Nivel_de_Ansiedade",
        "addiction_level": "Nivel_de_Vicio",
        "risk_depression": "Risco_Depressao",
    }

    # 🌟 2. O seu dicionário para os textos internos do gráfico
    traducao_valores = {
        "squared_error": "erro_quadratico",
        "samples": "amostras",
        "value": "valor_medio",
    }

    # Renomeia as colunas no DataFrame
    df = df.rename(columns=traducao_colunas)

    # Converter colunas numéricas
    colunas_numericas = [
        "Horas_Redes_Sociais_Diarias",
        "Horas_de_Sono",
        "Desempenho_Academico",
        "Atividade_Fisica_Horas",
        "Nivel_de_Ansiedade",
        "Nivel_de_Vicio",
        "Nivel_de_Estresse",
    ]

    for coluna in colunas_numericas:
        df[coluna] = df[coluna].astype(str).str.replace(",", ".", regex=False)
        df[coluna] = pd.to_numeric(df[coluna], errors="coerce")

    # Tratar coluna categórica
    df["Nivel_Interacao_Social"] = (
        df["Nivel_Interacao_Social"].astype(str).str.strip().str.lower()
    )
    df["Nivel_Interacao_Social"] = df["Nivel_Interacao_Social"].map(
        {"low": 1, "medium": 2, "high": 3}
    )

    # Remover nulos
    colunas_modelo = colunas_numericas + ["Nivel_Interacao_Social"]
    df = df.dropna(subset=colunas_modelo)

    # 🌟 O SEGREDO: A função agora devolve o DataFrame E as traduções do gráfico juntos
    return df, traducao_valores