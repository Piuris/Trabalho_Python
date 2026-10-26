"""Gráficos exploratórios do dataset (bons para a apresentação).

Uso:
    python visualizar_dados.py

Gera em relatorios/:
  - amostras_por_letra.png : balanceamento do dataset
  - maos_medias.png        : a "mão média" de cada letra depois da normalização
  - pca.png                : as 91 características projetadas em 2D (PCA)
"""
import argparse
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import seaborn as sns  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from config import ARQUIVO_DADOS, PASTA_RELATORIOS  # noqa: E402
from mao import CONEXOES, normalizar  # noqa: E402
from treinar import COLUNAS_PONTOS, carregar  # noqa: E402


def grafico_amostras(df, caminho):
    plt.figure(figsize=(10, 4))
    ordem = sorted(df["letra"].unique())
    sns.countplot(data=df, x="letra", hue="pessoa", order=ordem)
    plt.title("Amostras por letra")
    plt.tight_layout()
    plt.savefig(caminho, dpi=150)
    plt.close()


def grafico_maos_medias(df, caminho):
    letras = sorted(df["letra"].unique())
    colunas = 7
    linhas = int(np.ceil(len(letras) / colunas))
    fig, eixos = plt.subplots(linhas, colunas, figsize=(colunas * 2, linhas * 2.3))
    for ax in np.ravel(eixos):
        ax.axis("off")
    for ax, letra in zip(np.ravel(eixos), letras):
        sub = df[df["letra"] == letra]
        pontos = sub[COLUNAS_PONTOS].to_numpy(np.float32).reshape(-1, 21, 3)
        media = np.mean([normalizar(p, lado) for p, lado in zip(pontos, sub["lado"])], axis=0)
        for i, j in CONEXOES:
            ax.plot(media[[i, j], 0], media[[i, j], 1], "-", color="tab:blue", lw=2)
        ax.scatter(media[:, 0], media[:, 1], s=8, color="tab:red", zorder=3)
        ax.set_title(letra, fontsize=14, fontweight="bold")
        ax.set_aspect("equal")
        ax.invert_yaxis()   # na imagem o y cresce para baixo
    fig.suptitle("Mão média de cada letra (após normalização)")
    plt.tight_layout()
    plt.savefig(caminho, dpi=150)
    plt.close()


def grafico_pca(df, X, caminho):
    Z = PCA(n_components=2, random_state=42).fit_transform(StandardScaler().fit_transform(X))
    letras = sorted(df["letra"].unique())
    cores = sns.color_palette("husl", len(letras))
    plt.figure(figsize=(10, 8))
    for letra, cor in zip(letras, cores):
        m = (df["letra"] == letra).to_numpy()
        plt.scatter(Z[m, 0], Z[m, 1], s=4, alpha=0.4, color=cor)
        cx, cy = Z[m].mean(axis=0)
        plt.text(cx, cy, letra, fontsize=13, fontweight="bold", ha="center", va="center",
                 bbox={"boxstyle": "circle", "fc": "white", "ec": cor})
    plt.title("PCA das características (cada ponto é um frame)")
    plt.xlabel("componente 1")
    plt.ylabel("componente 2")
    plt.tight_layout()
    plt.savefig(caminho, dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dados", type=Path, default=ARQUIVO_DADOS)
    parser.add_argument("--relatorios", type=Path, default=PASTA_RELATORIOS)
    args = parser.parse_args()
    args.relatorios.mkdir(parents=True, exist_ok=True)

    df, X = carregar(args.dados)
    grafico_amostras(df, args.relatorios / "amostras_por_letra.png")
    grafico_maos_medias(df, args.relatorios / "maos_medias.png")
    grafico_pca(df, X, args.relatorios / "pca.png")
    print(f"Gráficos salvos em {args.relatorios}")


if __name__ == "__main__":
    main()
