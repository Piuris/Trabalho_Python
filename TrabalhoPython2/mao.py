import time
import urllib.request
from dataclasses import dataclass

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

from config import ARQUIVO_MODELO_MAO, URL_MODELO_MAO

CONEXOES = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]
PONTAS_DEDOS = [4, 8, 12, 16, 20]

ARTICULACOES = [
    (0, 1, 2), (1, 2, 3), (2, 3, 4),
    (0, 5, 6), (5, 6, 7), (6, 7, 8),
    (0, 9, 10), (9, 10, 11), (10, 11, 12),
    (0, 13, 14), (13, 14, 15), (14, 15, 16),
    (0, 17, 18), (17, 18, 19), (18, 19, 20),
]

PARES_PONTAS = [
    (PONTAS_DEDOS[i], PONTAS_DEDOS[j])
    for i in range(len(PONTAS_DEDOS))
    for j in range(i + 1, len(PONTAS_DEDOS))
]

REFERENCIAS_POLEGAR = [6, 7, 8]

N_COORDS = 21 * 3
_inicio_dist = N_COORDS
_inicio_ang = _inicio_dist + len(PARES_PONTAS)
_inicio_pol = _inicio_ang + len(ARTICULACOES)
GRUPOS_CARACTERISTICAS = {
    "coordenadas": slice(0, _inicio_dist),
    "distancias": slice(_inicio_dist, _inicio_ang),
    "angulos": slice(_inicio_ang, _inicio_pol),
    "polegar": slice(_inicio_pol, _inicio_pol + len(REFERENCIAS_POLEGAR)),
}


@dataclass
class Mao:
    pontos: np.ndarray
    lado: str
    proporcao: float

    def pontos_corrigidos(self) -> np.ndarray:
        p = self.pontos.copy()
        p[:, 0] *= self.proporcao
        p[:, 2] *= self.proporcao
        return p


def baixar_modelo_mao():
    if ARQUIVO_MODELO_MAO.exists():
        return
    ARQUIVO_MODELO_MAO.parent.mkdir(parents=True, exist_ok=True)
    print(f"Baixando modelo do MediaPipe em {URL_MODELO_MAO} ...")
    urllib.request.urlretrieve(URL_MODELO_MAO, ARQUIVO_MODELO_MAO)


class DetectorMao:

    def __init__(self, confianca: float = 0.6):
        baixar_modelo_mao()
        opcoes = vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(
                model_asset_buffer=ARQUIVO_MODELO_MAO.read_bytes()
            ),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=confianca,
            min_hand_presence_confidence=confianca,
            min_tracking_confidence=0.5,
        )
        self._detector = vision.HandLandmarker.create_from_options(opcoes)
        self._inicio = time.monotonic()
        self._ultimo_ms = -1

    def detectar(self, frame_bgr: np.ndarray) -> Mao | None:
        altura, largura = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        imagem = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        ms = int((time.monotonic() - self._inicio) * 1000)
        ms = max(ms, self._ultimo_ms + 1)
        self._ultimo_ms = ms

        resultado = self._detector.detect_for_video(imagem, ms)
        if not resultado.hand_landmarks:
            return None
        pontos = np.array(
            [[p.x, p.y, p.z] for p in resultado.hand_landmarks[0]], dtype=np.float32
        )
        lado = resultado.handedness[0][0].category_name
        return Mao(pontos=pontos, lado=lado, proporcao=largura / altura)

    def fechar(self):
        self._detector.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.fechar()


def normalizar(pontos: np.ndarray, lado: str) -> np.ndarray:
    p = pontos.astype(np.float32).copy()
    p -= p[0]
    if lado == "Left":
        p[:, 0] *= -1
    tamanho_palma = np.linalg.norm(p[9])
    return p / (tamanho_palma + 1e-8)


def _angulos(p: np.ndarray) -> np.ndarray:
    a = np.array([p[i] for i, _, _ in ARTICULACOES])
    b = np.array([p[j] for _, j, _ in ARTICULACOES])
    c = np.array([p[k] for _, _, k in ARTICULACOES])
    v1, v2 = a - b, c - b
    cos = np.sum(v1 * v2, axis=1) / (
        np.linalg.norm(v1, axis=1) * np.linalg.norm(v2, axis=1) + 1e-8
    )
    return np.arccos(np.clip(cos, -1.0, 1.0))


def _profundidade_polegar(p: np.ndarray) -> np.ndarray:
    normal = np.cross(p[5] - p[0], p[17] - p[0])
    normal /= np.linalg.norm(normal) + 1e-8
    return np.array([np.dot(p[4] - p[k], normal) for k in REFERENCIAS_POLEGAR])


def extrair_caracteristicas(pontos: np.ndarray, lado: str) -> np.ndarray:
    p = normalizar(pontos, lado)
    distancias = [np.linalg.norm(p[i] - p[j]) for i, j in PARES_PONTAS]
    return np.concatenate(
        [p.ravel(), distancias, _angulos(p), _profundidade_polegar(p)]
    ).astype(np.float32)


def desenhar_mao(frame: np.ndarray, mao: Mao, cor=(0, 255, 0)):
    altura, largura = frame.shape[:2]
    px = [(int(x * largura), int(y * altura)) for x, y, _ in mao.pontos]
    for i, j in CONEXOES:
        cv2.line(frame, px[i], px[j], cor, 2)
    for i, (x, y) in enumerate(px):
        raio = 6 if i in PONTAS_DEDOS else 4
        cv2.circle(frame, (x, y), raio, (255, 255, 255), -1)
        cv2.circle(frame, (x, y), raio, cor, 1)
