"""Configurações centrais do projeto (caminhos, letras e parâmetros)."""
import sys
from pathlib import Path

# O console do Windows pode usar cp1252, que quebra com acentos e setas.
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

# Letras do alfabeto manual de Libras que são ESTÁTICAS (uma pose só).
# H, J, K, X, Z e Ç exigem movimento e ficam de fora do classificador
# frame a frame — ver "Limitações" no README.
LETRAS = list("ABCDEFGILMNOPQRSTUVWY")

INDICE_CAMERA = 0
AMOSTRAS_POR_GRAVACAO = 150

# Tradutor em tempo real
JANELA_SUAVIZACAO = 10       # nº de frames considerados na votação
FRACAO_VOTOS = 0.7           # fração mínima de votos para a letra ser "estável"
CONFIANCA_MINIMA = 0.6       # probabilidade mínima para um frame votar
TEMPO_CONFIRMACAO = 0.8      # segundos segurando a letra estável para digitá-la
