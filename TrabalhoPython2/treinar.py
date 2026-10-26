"""Treina e compara classificadores para as letras de Libras.

Uso:
    python treinar.py

O que o script faz:
  1. Lê dados/landmarks.csv e transforma cada frame em 91 características (mao.py).
  2. Compara 4 modelos (KNN, SVM, Random Forest, MLP) com dois tipos de validação:
       - split ALEATÓRIO: frames vizinhos da mesma gravação caem no treino e no
         teste. Como são quase idênticos, o resultado fica otimista demais
         (vazamento de dados).
       - split POR SESSÃO: uma gravação inteira fica só no treino ou só no
         teste. É uma estimativa mais honesta do desempenho em uso real.
  3. Se houver mais de uma pessoa: leave-one-person-out (treina com todas
     menos uma e testa na que ficou de fora). Mede se o modelo serve para
     alguém que ele nunca viu.
  4. Ablação: quanto cada grupo de características (coordenadas, distâncias,
     ângulos) contribui.
  5. Gera matriz de confusão e relatório em relatorios/, e salva o melhor
     modelo (treinado com todos os dados) em modelos/classificador.joblib.
"""
import argparse
import time
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import seaborn as sns  # noqa: E402
from sklearn.calibration import CalibratedClassifierCV  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.metrics import classification_report, confusion_matrix  # noqa: E402
from sklearn.model_selection import (  # noqa: E402
    LeaveOneGroupOut,
    StratifiedGroupKFold,
    StratifiedKFold,
    cross_val_predict,
    cross_validate,
)
from sklearn.neighbors import KNeighborsClassifier  # noqa: E402
from sklearn.neural_network import MLPClassifier  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.svm import SVC  # noqa: E402

from config import ARQUIVO_CLASSIFICADOR, ARQUIVO_DADOS, PASTA_RELATORIOS  # noqa: E402
from mao import GRUPOS_CARACTERISTICAS, extrair_caracteristicas  # noqa: E402

COLUNAS_PONTOS = [f"{eixo}{i}" for i in range(21) for eixo in "xyz"]


def carregar(caminho: Path):
    df = pd.read_csv(caminho)
    pontos = df[COLUNAS_PONTOS].to_numpy(np.float32).reshape(-1, 21, 3)
    X = np.stack([extrair_caracteristicas(p, lado) for p, lado in zip(pontos, df["lado"])])
    return df, X


def criar_modelos():
    # KNN, SVM e MLP dependem de distâncias/gradientes → precisam de
    # padronização (média 0, desvio 1). Árvores não ligam para escala.
    return {
        "KNN": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5)),
        # O SVM não gera probabilidades por si só; o CalibratedClassifierCV
        # ajusta uma sigmoide sobre as distâncias à margem para obtê-las
        # (o tradutor precisa de probabilidades para o limiar de confiança).
        "SVM (RBF)": make_pipeline(
            StandardScaler(),
            CalibratedClassifierCV(SVC(C=10, gamma="scale"), ensemble=False),
        ),
        "Random Forest": RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=42),
        "MLP": make_pipeline(
            StandardScaler(),
            MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=1000,
                          early_stopping=True, random_state=42),
        ),
    }


def validacao_por_sessao(y, sessoes):
    sessoes_por_letra = pd.Series(sessoes).groupby(np.asarray(y)).nunique().min()
    if sessoes_por_letra < 2:
        print("AVISO: alguma letra tem só 1 gravação; não dá para separar por sessão.\n"
              "       Usando split aleatório (resultado otimista). Grave cada letra 2+ vezes.")
        return StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    return StratifiedGroupKFold(n_splits=min(5, sessoes_por_letra), shuffle=True,
                                random_state=42)


def comparar_modelos(X, y, sessoes, cv_sessao):
    cv_aleatorio = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    linhas = []
    for nome, modelo in criar_modelos().items():
        inicio = time.perf_counter()
        aleatorio = cross_validate(modelo, X, y, cv=cv_aleatorio, scoring="accuracy", n_jobs=-1)
        sessao = cross_validate(modelo, X, y, cv=cv_sessao, groups=sessoes,
                                scoring=["accuracy", "f1_macro"], n_jobs=-1)
        linhas.append({
            "modelo": nome,
            "acc_aleatorio": aleatorio["test_score"].mean(),
            "acc_sessao": sessao["test_accuracy"].mean(),
            "acc_sessao_std": sessao["test_accuracy"].std(),
            "f1_sessao": sessao["test_f1_macro"].mean(),
            "tempo_s": time.perf_counter() - inicio,
        })
        print(f"  {nome:<14} aleatório={linhas[-1]['acc_aleatorio']:.3f}  "
              f"por sessão={linhas[-1]['acc_sessao']:.3f}  f1={linhas[-1]['f1_sessao']:.3f}")
    return pd.DataFrame(linhas).sort_values("f1_sessao", ascending=False)


def avaliar_por_pessoa(modelo, X, y, pessoas):
    resultado = {}
    logo = LeaveOneGroupOut()
    for treino, teste in logo.split(X, y, groups=pessoas):
        pessoa = pessoas[teste[0]]
        modelo.fit(X[treino], y[treino])
        resultado[pessoa] = (modelo.predict(X[teste]) == y[teste]).mean()
    return resultado


def ablacao(modelo, X, y, sessoes, cv):
    g = GRUPOS_CARACTERISTICAS
    combinacoes = {
        "só coordenadas": [g["coordenadas"]],
        "só distâncias": [g["distancias"]],
        "só ângulos": [g["angulos"]],
        "distâncias + ângulos": [g["distancias"], g["angulos"]],
        "todas menos polegar": [g["coordenadas"], g["distancias"], g["angulos"]],
        "todas": [g["coordenadas"], g["distancias"], g["angulos"], g["polegar"]],
    }
    resultado = {}
    for nome, fatias in combinacoes.items():
        colunas = np.concatenate([np.arange(X.shape[1])[f] for f in fatias])
        r = cross_validate(modelo, X[:, colunas], y, cv=cv, groups=sessoes,
                           scoring="accuracy", n_jobs=-1)
        resultado[nome] = (len(colunas), r["test_score"].mean())
    return resultado


def grafico_comparacao(tabela, caminho):
    dados = tabela.melt(id_vars="modelo", value_vars=["acc_aleatorio", "acc_sessao"],
                        var_name="validação", value_name="acurácia")
    dados["validação"] = dados["validação"].map(
        {"acc_aleatorio": "split aleatório (otimista)", "acc_sessao": "split por sessão (realista)"})
    plt.figure(figsize=(9, 5))
    ax = sns.barplot(data=dados, x="modelo", y="acurácia", hue="validação")
    for barra in ax.containers:
        ax.bar_label(barra, fmt="%.3f", fontsize=8)
    ax.set_ylim(max(0, dados["acurácia"].min() - 0.1), 1.0)
    ax.legend(title="validação", loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2)
    ax.set_title("Comparação de modelos")
    plt.tight_layout()
    plt.savefig(caminho, dpi=150)
    plt.close()


def grafico_confusao(y_real, y_prev, letras, nome, caminho):
    matriz = confusion_matrix(y_real, y_prev, labels=letras, normalize="true")
    rotulos = np.where(matriz > 0.005, (matriz * 100).round().astype(int).astype(str), "")
    plt.figure(figsize=(12, 10))
    sns.heatmap(matriz, annot=rotulos, fmt="", cmap="Blues", xticklabels=letras,
                yticklabels=letras, cbar_kws={"label": "fração da letra real"},
                annot_kws={"fontsize": 7})
    plt.xlabel("letra prevista")
    plt.ylabel("letra real")
    plt.title(f"Matriz de confusão (%) — {nome}, validação por sessão")
    plt.tight_layout()
    plt.savefig(caminho, dpi=150)
    plt.close()
    return matriz


def maiores_confusoes(matriz, letras, n=10):
    pares = [(matriz[i, j], letras[i], letras[j])
             for i in range(len(letras)) for j in range(len(letras))
             if i != j and matriz[i, j] > 0]
    return sorted(pares, reverse=True)[:n]


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dados", type=Path, default=ARQUIVO_DADOS)
    parser.add_argument("--modelo", type=Path, default=ARQUIVO_CLASSIFICADOR)
    parser.add_argument("--relatorios", type=Path, default=PASTA_RELATORIOS)
    args = parser.parse_args()
    args.relatorios.mkdir(parents=True, exist_ok=True)
    args.modelo.parent.mkdir(parents=True, exist_ok=True)

    df, X = carregar(args.dados)
    y = df["letra"].to_numpy()
    sessoes = df["sessao"].to_numpy()
    pessoas = df["pessoa"].to_numpy()
    letras = sorted(np.unique(y))
    print(f"{len(df)} amostras | {len(letras)} letras | {df['sessao'].nunique()} sessões | "
          f"{df['pessoa'].nunique()} pessoa(s) | {X.shape[1]} características\n")

    cv_sessao = validacao_por_sessao(y, sessoes)

    print("Comparando modelos (validação cruzada):")
    tabela = comparar_modelos(X, y, sessoes, cv_sessao)
    grafico_comparacao(tabela, args.relatorios / "comparacao_modelos.png")

    melhor_nome = tabela.iloc[0]["modelo"]
    melhor = criar_modelos()[melhor_nome]
    print(f"\nMelhor modelo (F1 macro por sessão): {melhor_nome}")

    y_prev = cross_val_predict(melhor, X, y, cv=cv_sessao, groups=sessoes, n_jobs=-1)
    matriz = grafico_confusao(y, y_prev, letras, melhor_nome,
                              args.relatorios / "matriz_confusao.png")
    confusoes = maiores_confusoes(matriz, letras)

    print("\nAblação das características:")
    abl = ablacao(melhor, X, y, sessoes, cv_sessao)
    for nome, (n_col, acc) in abl.items():
        print(f"  {nome:<22} ({n_col:>2} colunas)  acurácia={acc:.3f}")

    por_pessoa = {}
    if len(np.unique(pessoas)) >= 2:
        print("\nLeave-one-person-out:")
        por_pessoa = avaliar_por_pessoa(criar_modelos()[melhor_nome], X, y, pessoas)
        for pessoa, acc in por_pessoa.items():
            print(f"  testando em {pessoa:<12} acurácia={acc:.3f}")

    # Relatório em texto
    with open(args.relatorios / "relatorio.txt", "w", encoding="utf-8") as f:
        f.write(f"Amostras: {len(df)} | Letras: {len(letras)} | Sessões: {df['sessao'].nunique()} "
                f"| Pessoas: {df['pessoa'].nunique()}\n\n")
        f.write("== Comparação de modelos ==\n")
        f.write(tabela.to_string(index=False, float_format="%.3f") + "\n\n")
        f.write(f"== Relatório por letra — {melhor_nome} (validação por sessão) ==\n")
        f.write(classification_report(y, y_prev, digits=3) + "\n")
        f.write("== Pares mais confundidos (real → previsto: % da letra real) ==\n")
        for frac, real, prev in confusoes:
            f.write(f"  {real} → {prev}: {frac * 100:.1f}%\n")
        f.write("\n== Ablação das características ==\n")
        for nome, (n_col, acc) in abl.items():
            f.write(f"  {nome:<22} ({n_col:>2} colunas)  acurácia={acc:.3f}\n")
        if por_pessoa:
            f.write("\n== Leave-one-person-out ==\n")
            for pessoa, acc in por_pessoa.items():
                f.write(f"  testando em {pessoa:<12} acurácia={acc:.3f}\n")

    print("\nPares mais confundidos:")
    for frac, real, prev in confusoes[:5]:
        print(f"  {real} → {prev}: {frac * 100:.1f}%")

    # Modelo final: treinado com TODOS os dados (a validação acima só serviu
    # para escolher o modelo e estimar o desempenho).
    melhor.fit(X, y)
    joblib.dump({"modelo": melhor, "nome": melhor_nome, "letras": list(melhor.classes_)},
                args.modelo)
    print(f"\nModelo salvo em {args.modelo}")
    print(f"Gráficos e relatório em {args.relatorios}")


if __name__ == "__main__":
    main()
