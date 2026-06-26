# main.py
from abordagem_estresse import executar_analise_estresse
from vicio_de_notas import executar_analise_vicio_de_notas


def iniciar_sistema():
    print("=======================================================")
    print("   SISTEMA DE ANÁLISE DE SAÚDE MENTAL E PERFORMANCE    ")
    print("=======================================================")

    # 1. Abre a Árvore de Decisão (sem bloquear o terminal)
    executar_analise_estresse()

    # 2. Executa o Naïve Bayes e Apriori imediatamente na sequência
    executar_analise_vicio_de_notas()


if __name__ == "__main__":
    iniciar_sistema()