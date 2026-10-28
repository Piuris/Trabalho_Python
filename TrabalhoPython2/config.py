
import sys
from pathlib import Path

for _fluxo in (sys.stdout, sys.stderr):
    if hasattr(_fluxo, "reconfigure"):
        _fluxo.reconfigure(encoding="utf-8")

RAIZ = Path(__file__).resolve().parent

PASTA_DADOS = RAIZ / "dados"
ARQUIVO_DADOS = PASTA_DADOS / "landmarks.csv"

PASTA_MODELOS = RAIZ / "modelos"
ARQUIVO_CLASSIFICADOR = PASTA_MODELOS / "classificador.joblib"
ARQUIVO_MODELO_MAO = PASTA_MODELOS / "hand_landmarker.task"
URL_MODELO_MAO = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/latest/hand_landmarker.task"
)

PASTA_RELATORIOS = RAIZ / "relatorios"

LETRAS = list("ABCDEFGILMNOPQRSTUVWY")

INDICE_CAMERA = 0
AMOSTRAS_POR_GRAVACAO = 150

JANELA_SUAVIZACAO = 10
FRACAO_VOTOS = 0.7
CONFIANCA_MINIMA = 0.6
TEMPO_CONFIRMACAO = 0.8
