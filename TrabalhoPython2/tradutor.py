import argparse
import time
from collections import Counter, deque

import cv2
import joblib
import numpy as np

from config import (
    ARQUIVO_CLASSIFICADOR,
    CONFIANCA_MINIMA,
    FRACAO_VOTOS,
    INDICE_CAMERA,
    JANELA_SUAVIZACAO,
    TEMPO_CONFIRMACAO,
)
from mao import DetectorMao, desenhar_mao, extrair_caracteristicas


def escrever(frame, texto, pos, escala=0.7, cor=(255, 255, 255), espessura=2):
    cv2.putText(frame, texto, pos, cv2.FONT_HERSHEY_SIMPLEX, escala, (0, 0, 0),
                espessura + 3, cv2.LINE_AA)
    cv2.putText(frame, texto, pos, cv2.FONT_HERSHEY_SIMPLEX, escala, cor,
                espessura, cv2.LINE_AA)


def desenhar_top3(frame, letras, probabilidades):
    x0, y0, largura_max = frame.shape[1] - 230, 30, 150
    for k, i in enumerate(np.argsort(probabilidades)[::-1][:3]):
        y = y0 + k * 35
        p = probabilidades[i]
        cor = (0, 200, 0) if p >= CONFIANCA_MINIMA else (0, 140, 255)
        cv2.rectangle(frame, (x0 + 40, y - 18), (x0 + 40 + int(largura_max * p), y + 4), cor, -1)
        cv2.rectangle(frame, (x0 + 40, y - 18), (x0 + 40 + largura_max, y + 4), (255, 255, 255), 1)
        escrever(frame, letras[i], (x0, y), 0.8)
        escrever(frame, f"{p:.0%}", (x0 + 45, y), 0.5, espessura=1)


def desenhar_texto(frame, texto):
    altura, largura = frame.shape[:2]
    cv2.rectangle(frame, (0, altura - 70), (largura, altura), (0, 0, 0), -1)
    cursor = "|" if int(time.time() * 2) % 2 else " "
    escrever(frame, texto[-30:] + cursor, (20, altura - 25), 1.2, (255, 255, 255), 2)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--camera", type=int, default=INDICE_CAMERA)
    args = parser.parse_args()

    if not ARQUIVO_CLASSIFICADOR.exists():
        raise SystemExit("Modelo não encontrado. Rode antes: python coletar.py e python treinar.py")
    salvo = joblib.load(ARQUIVO_CLASSIFICADOR)
    modelo, letras = salvo["modelo"], salvo["letras"]
    print(f"Modelo carregado: {salvo['nome']} ({len(letras)} letras)")

    camera = cv2.VideoCapture(args.camera)
    if not camera.isOpened():
        raise SystemExit(f"Não consegui abrir a câmera {args.camera}")

    votos = deque(maxlen=JANELA_SUAVIZACAO)
    texto = ""
    candidata, inicio_candidata = None, 0.0
    ultima_digitada = None
    ultimo_tempo, fps = time.perf_counter(), 0.0

    with DetectorMao() as detector:
        while True:
            ok, frame = camera.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)
            mao = detector.detectar(frame)

            if mao:
                desenhar_mao(frame, mao)
                x = extrair_caracteristicas(mao.pontos_corrigidos(), mao.lado)
                probabilidades = modelo.predict_proba(x.reshape(1, -1))[0]
                melhor = int(np.argmax(probabilidades))
                votos.append(letras[melhor] if probabilidades[melhor] >= CONFIANCA_MINIMA else None)
                desenhar_top3(frame, letras, probabilidades)
            else:
                votos.append(None)
            estavel = None
            if len(votos) == votos.maxlen:
                letra, n = Counter(votos).most_common(1)[0]
                if letra is not None and n >= FRACAO_VOTOS * votos.maxlen:
                    estavel = letra

            agora = time.monotonic()
            if estavel != candidata:
                candidata, inicio_candidata = estavel, agora
            if estavel is None:
                if not mao:
                    ultima_digitada = None
            elif estavel != ultima_digitada:
                progresso = min(1.0, (agora - inicio_candidata) / TEMPO_CONFIRMACAO)
                escrever(frame, estavel, (30, 110), 3.0, (0, 255, 255), 5)
                cv2.rectangle(frame, (30, 130), (30 + int(120 * progresso), 145), (0, 255, 255), -1)
                cv2.rectangle(frame, (30, 130), (150, 145), (255, 255, 255), 1)
                if progresso >= 1.0:
                    texto += estavel
                    ultima_digitada = estavel
            else:
                escrever(frame, estavel, (30, 110), 3.0, (0, 200, 0), 5)

            t = time.perf_counter()
            fps = 0.9 * fps + 0.1 / max(t - ultimo_tempo, 1e-6)
            ultimo_tempo = t
            escrever(frame, f"{fps:.0f} FPS | {salvo['nome']}", (20, 30), 0.55, espessura=1)

            desenhar_texto(frame, texto)
            cv2.imshow("Tradutor de Libras", frame)

            tecla = cv2.waitKey(1) & 0xFF
            if tecla == 27:
                break
            if tecla == ord(" "):
                texto += " "
                ultima_digitada = None
            elif tecla == 8:
                texto = texto[:-1]
                ultima_digitada = None
            elif tecla == 13:
                texto = ""
                ultima_digitada = None

    camera.release()
    cv2.destroyAllWindows()
    print(f"Texto final: {texto}")


if __name__ == "__main__":
    main()
