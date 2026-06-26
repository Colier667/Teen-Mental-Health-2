import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import plot_tree
from sklearn.metrics import mean_absolute_error, r2_score

# ==========================
# 1. Carregar dataset
# ==========================

df = pd.read_csv("Teen_Mental_Health_Dataset.csv", sep=";")

print("Quantidade de linhas original:", len(df))

# ==========================
# 2. Converter colunas numéricas
# ==========================

colunas_numericas = [
    "daily_social_media_hours",
    "sleep_hours",
    "academic_performance",
    "physical_activity",
    "anxiety_level",
    "addiction_level",
    "stress_level"
]

for coluna in colunas_numericas:

    print(f"\nVerificando coluna: {coluna}")
    print(df[coluna].head())

    df[coluna] = (
        df[coluna]
        .astype(str)
        .str.replace(",", ".", regex=False)
    )

    df[coluna] = pd.to_numeric(df[coluna], errors="coerce")

# ==========================
# 3. Converter coluna categórica
# ==========================

print("\nValores encontrados em social_interaction_level:")
print(df["social_interaction_level"].unique())

df["social_interaction_level"] = (
    df["social_interaction_level"]
    .astype(str)
    .str.strip()
    .str.lower()
)

df["social_interaction_level"] = df["social_interaction_level"].map({
    "low": 1,
    "medium": 2,
    "high": 3
})

# ==========================
# 4. Conferir valores nulos
# ==========================

print("\nValores nulos por coluna:")

colunas_modelo = [
    "daily_social_media_hours",
    "sleep_hours",
    "academic_performance",
    "physical_activity",
    "social_interaction_level",
    "anxiety_level",
    "addiction_level",
    "stress_level"
]

print(df[colunas_modelo].isnull().sum())

df = df.dropna(subset=colunas_modelo)

print("\nQuantidade de linhas após tratamento:", len(df))

if len(df) == 0:
    print("\nERRO: Todas as linhas foram removidas.")
    exit()

# ==========================
# 5. Variáveis do modelo
# ==========================

X = df[
    [
        "daily_social_media_hours",
        "sleep_hours",
        "academic_performance",
        "physical_activity",
        "social_interaction_level",
        "anxiety_level",
        "addiction_level"
    ]
]

y = df["stress_level"]

print("\nFormato de X:", X.shape)
print("Formato de y:", y.shape)

# ==========================
# 6. Treino e teste
# ==========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ==========================
# 7. Árvore de Decisão
# ==========================

modelo = DecisionTreeRegressor(
    max_depth=4,
    random_state=42
)

modelo.fit(X_train, y_train)

# ==========================
# 8. Previsões
# ==========================

predicoes = modelo.predict(X_test)

print("\n===== RESULTADOS =====")
print("R²:", r2_score(y_test, predicoes))
print("Erro Médio Absoluto:", mean_absolute_error(y_test, predicoes))

# ==========================
# 9. Desenhar árvore
# ==========================

plt.figure(figsize=(18, 10))

plot_tree(
    modelo,
    feature_names=X.columns,
    filled=True
)

plt.title("Árvore de Decisão - Previsão do Nível de Estresse")

plt.show()