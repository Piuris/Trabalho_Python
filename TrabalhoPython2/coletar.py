import argparse
import csv
import time
from collections import Counter

import cv2
import pandas as pd

from config import AMOSTRAS_POR_GRAVACAO, ARQUIVO_DADOS, INDICE_CAMERA, LETRAS
from mao import DetectorMao, desenhar_mao

COLUNAS = ["letra", "pessoa", "sessao", "lado"] + [
    f"{eixo}{i}" for i in range(21) for eixo in "xyz"
]
CONTAGEM_REGRESSIVA = 2.0


def carregar_contagem() -> Counter:
    if not ARQUIVO_DADOS.exists():
        return Counter()
    return Counter(pd.read_csv(ARQUIVO_DADOS, usecols=["letra"])["letra"])


def escrever(frame, texto, pos, escala=0.7, cor=(255, 255, 255), espessura=2):
    cv2.putText(frame, texto, pos, cv2.FONT_HERSHEY_SIMPLEX, escala, (0, 0, 0),
                espessura + 3, cv2.LINE_AA)
    cv2.putText(frame, texto, pos, cv2.FONT_HERSHEY_SIMPLEX, escala, cor,
                espessura, cv2.LINE_AA)


def desenhar_painel(frame, contagem, letra_atual):
    x0, y = frame.shape[1] - 130, 25
    for letra in LETRAS:
        n = contagem.get(letra, 0)
        cor = (0, 255, 255) if letra == letra_atual else (
            (0, 255, 0) if n >= 300 else (200, 200, 200))
        escrever(frame, f"{letra}: {n}", (x0, y), 0.5, cor, 1)
        y += 20


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pessoa", required=True, help="nome de quem está gravando")
    parser.add_argument("--amostras", type=int, default=AMOSTRAS_POR_GRAVACAO)
    parser.add_argument("--camera", type=int, default=INDICE_CAMERA)
    args = parser.parse_args()

    ARQUIVO_DADOS.parent.mkdir(parents=True, exist_ok=True)
    novo_arquivo = not ARQUIVO_DADOS.exists()
    arquivo = open(ARQUIVO_DADOS, "a", newline="", encoding="utf-8")
    escritor = csv.writer(arquivo)
    if novo_arquivo:
        escritor.writerow(COLUNAS)

    contagem = carregar_contagem()
    letra_atual = None
    gravando = False
    inicio_gravacao = 0.0
    restantes = 0
    sessao = ""

    camera = cv2.VideoCapture(args.camera)
    if not camera.isOpened():
        raise SystemExit(f"Não consegui abrir a câmera {args.camera}")

    with DetectorMao() as detector:
        while True:
            ok, frame = camera.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)
            mao = detector.detectar(frame)
            if mao:
                desenhar_mao(frame, mao)

            if gravando:
                falta = CONTAGEM_REGRESSIVA - (time.monotonic() - inicio_gravacao)
                if falta > 0:
                    escrever(frame, f"Prepare-se: {falta:.1f}s", (20, 80), 1.0, (0, 255, 255))
                elif mao:
                    pontos = mao.pontos_corrigidos()
                    escritor.writerow([letra_atual, args.pessoa, sessao, mao.lado]
                                      + [f"{v:.6f}" for v in pontos.ravel()])
                    contagem[letra_atual] += 1
                    restantes -= 1
                    if restantes == 0:
                        gravando = False
                        arquivo.flush()
                if gravando and falta <= 0:
                    escrever(frame, f"GRAVANDO {letra_atual}  faltam {restantes}",
                             (20, 80), 0.9, (0, 0, 255))
                    escrever(frame, "mexa um pouco a mao: perto/longe, inclinada, lados",
                             (20, 110), 0.55, (0, 0, 255), 1)

            escrever(frame, f"Pessoa: {args.pessoa}   Letra: {letra_atual or '-'}", (20, 35))
            if not mao:
                escrever(frame, "nenhuma mao detectada", (20, frame.shape[0] - 50),
                         0.6, (0, 0, 255))
            escrever(frame, "letra = escolher | ESPACO = gravar | ESC = sair",
                     (20, frame.shape[0] - 20), 0.55, (255, 255, 255), 1)
            desenhar_painel(frame, contagem, letra_atual)
            cv2.imshow("Coleta - Libras", frame)

            tecla = cv2.waitKey(1) & 0xFF
            if tecla == 27:
                break
            if tecla == ord(" ") and letra_atual and not gravando:
                gravando = True
                inicio_gravacao = time.monotonic()
                restantes = args.amostras
                sessao = f"{args.pessoa}_{letra_atual}_{int(time.time())}"
            elif 0 < tecla < 128 and chr(tecla).upper() in LETRAS and not gravando:
                letra_atual = chr(tecla).upper()

    arquivo.close()
    camera.release()
    cv2.destroyAllWindows()
    print(f"Dados salvos em {ARQUIVO_DADOS}")


if __name__ == "__main__":
    main()
