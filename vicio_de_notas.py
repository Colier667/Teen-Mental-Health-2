import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt

# Importa a função de tratamento unificado do grupo
from tratamento_de_dados import carregar_e_tratar_dados

def executar_analise_vicio_de_notas():
    print("\nIniciando análises de Vício e Notas Escolares...")
    df, _ = carregar_e_tratar_dados()
    
    # Se a coluna 'Age' ou 'Idade' não foi traduzida no arquivo central, 
    # garantimos que estamos usando a coluna de idade correta do dataset original.
    # Caso no seu tratamento ela se chame 'Idade', mude aqui para 'Idade'.
    coluna_idade = "Age" if "Age" in df.columns else "Idade" if "Idade" in df.columns else df.columns[0]

    # ========================================================
    # 1. NAÏVE BAYES: Prever queda de nota por causa da Ansiedade
    # ========================================================
    # Definimos queda de nota (classe 1) se estiver abaixo da mediana do grupo
    mediana_nota = df["Desempenho_Academico"].median()
    df["target_nota_baixa"] = (df["Desempenho_Academico"] < mediana_nota).astype(int)
    
    # O foco estrito pedido: Ansiedade impactando a Nota
    X_nb = df[["Nivel_de_Ansiedade"]] 
    y_nb = df["target_nota_baixa"]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_nb, y_nb, test_size=0.2, random_state=42, stratify=y_nb
    )
    
    modelo_nb = GaussianNB()
    modelo_nb.fit(X_train, y_train)
    
    pred = modelo_nb.predict(X_test)
    acuracia = accuracy_score(y_test, pred)
    relatorio_texto = classification_report(y_test, pred, zero_division=0)
    
    # ========================================================
    # 2. APRIORI: Uso do TikTok vicia mais certas idades?
    # ========================================================
    df_apriori = pd.DataFrame()
    
    # Condições lógicas baseadas no enunciado
    df_apriori["Usa_Muito_TikTok"] = (df["Plataforma_Mais_Usada"].str.strip().str.lower() == "tiktok") & (df["Horas_Redes_Sociais_Diarias"] > 4)
    df_apriori["Alto_Vicio"] = df["Nivel_de_Vicio"] > 5.0
    
    # Mapeamento de idades (ex: se o aluno tem menos de 16 anos / faixa jovem do Ensino Médio)
    df_apriori["Idade_Abaixo_16"] = df[coluna_idade] <= 15
    df_apriori["Idade_Acima_15"] = df[coluna_idade] > 15
    
    frequent_itemsets = apriori(df_apriori, min_support=0.02, use_colnames=True)
    regras = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.4)
    
    # ========================================================
    # 3. GERAÇÃO DE ALERTAS RECOMENDADOS (Regra de Negócio Extrapolada)
    # ========================================================
    plataforma_top = df["Plataforma_Mais_Usada"].value_counts().idxmax().upper()
    
    alerta_palestras = f"🚨 ALERTA PREVENTIVO DE VÍCIO:\nRecomenda-se a aplicação urgente de palestras focadas na rede social '{plataforma_top}', identificada como o maior vetor de engajamento diário da base de estudantes analisada."
    alerta_reforco = "📚 ALERTA PEDAGÓGICO RECOMENDADO:\nCom base no modelo preditivo de Ansiedade vs. Notas, recomenda-se a instituição de ciclos de reforço escolar preventivo antes do período de provas para mitigar quedas de desempenho."

    # Print estruturado no terminal
    print(f"\n> Acurácia do Naïve Bayes: {acuracia:.2%}")
    print("\n> Principais Regras de Associação Encontradas:")
    print(regras[["antecedents", "consequents", "confidence"]].head(3))
    print(f"\n{alerta_palestras}\n")
    print(f"{alerta_reforco}\n")

    # ========================================================
    # 4. EXPORTAÇÃO DO RELATÓRIO EM PNG (Para os Slides)
    # ========================================================
    fig, ax = plt.subplots(figsize=(11, 9), facecolor='#121212')
    ax.axis('off')
    
    texto_png = "==================================================================\n"
    texto_png += "        RELATÓRIO DE INTELIGÊNCIA: VÍCIO E NOTAS ESCOLARES       \n"
    texto_png += "==================================================================\n\n"
    texto_png += f"-> ACURÁCIA PREDIÇÃO DE QUEDA DE NOTA (Naïve Bayes): {acuracia:.2%}\n\n"
    texto_png += "-> MATRIZ DE SUPORTE E CLASSIFICAÇÃO:\n"
    texto_png += f"{relatorio_texto}\n"
    texto_png += "------------------------------------------------------------------\n"
    texto_png += "-> REGRAS DE ASSOCIAÇÃO MINERADAS (Apriori):\n\n"
    
    if not regras.empty:
        for idx, row in regras.head(2).iterrows():
            ant = list(row['antecedents'])
            consq = list(row['consequents'])
            conf = row['confidence']
            texto_png += f" Regra {idx+1}: Se {ant} -> Então {consq}\n"
            texto_png += f"          Nível de Confiança Estatística: {conf:.2%}\n\n"
    else:
        texto_png += " Nenhuma associação cruzada acima de 40% de confiança.\n\n"
        
    texto_png += "------------------------------------------------------------------\n"
    texto_png += f"{alerta_palestras}\n\n"
    texto_png += f"{alerta_reforco}\n"
    texto_png += "=================================================================="

    ax.text(0.02, 0.98, texto_png, 
            transform=ax.transAxes, 
            fontsize=10.5, 
            fontfamily='monospace', 
            color='#ffffff', 
            va='top', 
            ha='left')
    
    plt.savefig("relatorio_vicio_notas.png", bbox_inches='tight', dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)
    print("💾 Novo relatório em PNG alinhado às regras do professor gerado com sucesso!")

if __name__ == "__main__":
    executar_analise_vicio_de_notas()