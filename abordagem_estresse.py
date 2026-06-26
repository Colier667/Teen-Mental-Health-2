import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, plot_tree

# Importa a função de tratamento unificado
from tratamento_de_dados import carregar_e_tratar_dados


# 🌟 ADICIONADO: A função que o main.py precisa chamar
def executar_analise_estresse():
    print("\n[Pessoa 3] - Iniciando análise da Árvore de Decisão (Estresse)...")

    # ========================================================
    # 1. CARREGAR E TRATAR DATASET (UNIFICADO)
    # ========================================================
    df, traduzir_grafico = carregar_e_tratar_dados()
    print(f"Quantidade de linhas após o tratamento unificado: {len(df)}")

    # ========================================================
    # 2. VARIÁVEIS DO MODELO (Utilizando os nomes traduzidos)
    # ========================================================
    X = df[
        [
            "Horas_Redes_Sociais_Diarias",
            "Horas_de_Sono",
            "Desempenho_Academico",
            "Atividade_Fisica_Horas",
            "Nivel_Interacao_Social",
            "Nivel_de_Ansiedade",
            "Nivel_de_Vicio",
        ]
    ]
    y = df["Nivel_de_Estresse"]

    print(f"\nFormato de X (Features): {X.shape}")
    print(f"Formato de y (Target): {y.shape}")

    # ========================================================
    # 3. DIVISÃO DE TREINO E TESTE
    # ========================================================
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # ========================================================
    # 4. MODELAGEM (ÁRVORE DE DECISÃO)
    # ========================================================
    modelo = DecisionTreeRegressor(max_depth=4, random_state=42)
    modelo.fit(X_train, y_train)

    # ========================================================
    # 5. PREVISÕES E MÉTRICAS
    # ========================================================
    predicoes = modelo.predict(X_test)

    print("\n" + "=" * 30)
    print("     RESULTADOS DO MODELO     ")
    print("=" * 30)
    print(f"R² (Coeficiente de Determinação): {r2_score(y_test, predicoes):.4f}")
    print(f"Erro Médio Absoluto (MAE):        {mean_absolute_error(y_test, predicoes):.4f}")
    print("=" * 30)

    # ========================================================
    # 6. DESENHAR ÁRVORE (ADAPTÁVEL, EM PORTUGUÊS E MAXIMIZADO)
    # ========================================================
    fig = plt.figure(figsize=(16, 9))

    elementos_texto = plot_tree(
        modelo,
        feature_names=list(X.columns),  # Mapeia os novos nomes em português nas caixinhas
        filled=True,
        rounded=True,
        fontsize=7,
        precision=2,
    )

    # Aplica a tradução que veio do tratamento centralizado para os termos internos
    for texto in elementos_texto:
        txt = texto.get_text()
        for ingles, portugues in traduzir_grafico.items():
            txt = txt.replace(ingles, portugues)
        texto.set_text(txt)

    plt.title(
        "Árvore de Decisão - Previsão do Nível de Estresse",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()

    # Força a janela a abrir em tela cheia no Windows
    try:
        fig_manager = plt.get_current_fig_manager()
        fig_manager.window.state("maximized")
    except Exception:
        pass

# Salva o gráfico perfeito e traduzido para o slide de vocês
    plt.savefig("arvore_estresse_auto.png", bbox_inches="tight", dpi=150)
    print("\n💾 Imagem traduzida salva com sucesso como 'arvore_estresse_auto.png'!")

    # Exibe o gráfico e segura a execução até o usuário fechar
    plt.show()


# Permite rodar esse arquivo sozinho se quiser testar só ele
if __name__ == "__main__":
    executar_analise_estresse()