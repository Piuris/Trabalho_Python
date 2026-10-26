# Tradutor de Libras — alfabeto manual em tempo real

Reconhece as letras estáticas do alfabeto manual de Libras pela webcam e forma palavras na tela.

```
webcam ──► MediaPipe Hands ──► 21 pontos (x,y,z) ──► normalização ──► 91 características ──► classificador ──► suavização ──► texto
            (rede neural pronta)                     (posição, escala,   (coords, distâncias,   (KNN / SVM /      (votação +
                                                      mão esq./dir.)      ângulos)               RF / MLP)          tempo)
```

## Instalação

Requer Python 3.10–3.12 (o MediaPipe ainda não tem suporte garantido para versões mais novas).

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows  (Linux/Mac: source .venv/bin/activate)
pip install -r requirements.txt
```

O modelo de mão do MediaPipe (`modelos/hand_landmarker.task`) é baixado sozinho na primeira execução.

## Como usar (passo a passo para replicar)

### 1. Coletar o dataset

```bash
python coletar.py --pessoa ana
```

- Aperte a **tecla da letra** (ex.: `a`) para escolhê-la.
- Aperte **ESPAÇO**. Após 2 s de contagem, ele grava 150 frames dessa letra.
- **Durante a gravação, mexa um pouco a mão**: mais perto, mais longe, mais inclinada, em várias partes da tela. A variedade no treino é o que deixa o modelo robusto.
- **Grave cada letra pelo menos 3 vezes** (sessões separadas). Se puder, peça para mais uma ou duas pessoas (amigos, família) gravarem com `--pessoa <nome>`. Isso habilita o teste leave-one-person-out no treino.
- No painel à direita aparece a quantidade de amostras por letra. Ela fica verde a partir de 300.

Tudo é salvo em `dados/landmarks.csv`, e novas gravações são acrescentadas ao final.

### 2. Explorar os dados (opcional, bom para os slides)

```bash
python visualizar_dados.py
```

### 3. Treinar e comparar os modelos

```bash
python treinar.py
```

O modelo escolhido vai para `modelos/classificador.joblib`, e os gráficos e o relatório vão para `relatorios/`.

### 4. Usar o tradutor

```bash
python tradutor.py
```

| Tecla | Ação |
|---|---|
| segurar uma letra | a barra amarela enche e a letra é digitada |
| ESPAÇO | espaço |
| BACKSPACE | apaga |
| ENTER | limpa |
| ESC | sai |

Para repetir uma letra ("SS"), tire a mão da câmera por um instante.

## Estrutura

| Arquivo | Função |
|---|---|
| `config.py` | caminhos, lista de letras e parâmetros do tradutor |
| `mao.py` | detector MediaPipe, normalização, extração das características e desenho |
| `coletar.py` | coleta do dataset pela webcam |
| `visualizar_dados.py` | balanceamento, "mão média" por letra e PCA |
| `treinar.py` | comparação dos modelos, validação, ablação, matriz de confusão e modelo final |
| `tradutor.py` | aplicação em tempo real |

## Decisões técnicas (o que explicar na apresentação)

### Por que landmarks, e não a imagem direto numa CNN?
O MediaPipe já resolve a parte difícil, que é achar a mão e os 21 pontos. Com isso o classificador recebe 91 números em vez de ~300 mil pixels. Com centenas de exemplos por letra, isso basta para treinar. Ele também fica imune a fundo, iluminação e cor da pele, e roda em tempo real na CPU.

### Normalização (`mao.normalizar`)
1. **Proporção do frame**: o MediaPipe divide x pela largura e y pela altura, então a mão sai "esticada" em frames não quadrados. Multiplicamos x e z por largura/altura.
2. **Translação**: o pulso vira a origem, e a posição na tela deixa de importar.
3. **Espelhamento**: na mão esquerda invertemos o x, e o mesmo modelo serve para destros e canhotos.
4. **Escala**: dividimos pelo comprimento da palma (pulso até a base do dedo médio), e a distância da câmera deixa de importar. A palma é usada porque não muda quando os dedos dobram.
5. **A rotação NÃO é normalizada**: em Libras a orientação faz parte do sinal.

### Características (`mao.extrair_caracteristicas`)
- **63 coordenadas** normalizadas
- **10 distâncias** entre as pontas dos dedos (ex.: o O e o F dependem do polegar tocando outro dedo)
- **15 ângulos** de articulação (o quanto cada dedo está dobrado; não mudam com rotação nem escala)
- **3 profundidades do polegar**: a ponta do polegar projetada na normal da palma, em relação a 3 pontos do indicador. Diz se o polegar está **na frente ou atrás** do indicador, que é a única diferença entre **F e T**. Ela foi criada depois que o modelo confundia as duas letras ao vivo: sem ela, ele separava F de T pela direção da palma (que era diferente nas gravações), e não pelo polegar. Por ser medida em relação à própria palma, não muda quando a mão gira.

O `treinar.py` faz uma **ablação** que mede quanto cada grupo contribui.

### Validação honesta: split por sessão
Frames consecutivos de uma mesma gravação são quase idênticos. Num split aleatório, o frame 41 cai no teste e os frames 40 e 42 no treino, e o modelo "cola". Isso é **vazamento de dados**. O `treinar.py` mostra as duas acurácias lado a lado: a do split **aleatório** (otimista) e a do split **por sessão** (`StratifiedGroupKFold`), em que uma gravação inteira fica só no treino ou só no teste. Com 2+ pessoas no dataset, ainda roda o **leave-one-person-out**: treina sem uma pessoa e testa nela, o que mede a generalização para usuários novos. Com uma pessoa só, essa etapa é pulada automaticamente.

### Modelos comparados
| Modelo | Ideia | Observação |
|---|---|---|
| KNN | vota com os 5 exemplos mais parecidos | simples, precisa de padronização |
| SVM (RBF) | fronteira de margem máxima num espaço não linear | probabilidades via calibração (`CalibratedClassifierCV`) |
| Random Forest | 300 árvores votando | não precisa de padronização, robusto |
| MLP | rede neural 91 → 128 → 64 → 21 | padronização + early stopping |

O vencedor (maior F1 macro na validação por sessão) é retreinado com todos os dados e salvo.

### Suavização em tempo real (`tradutor.py`)
Uma previsão por frame oscila, por isso a decisão passa por três filtros:
1. **confiança**: frames com probabilidade abaixo de 60% não votam;
2. **votação**: a letra precisa de 70% dos votos nos últimos 10 frames;
3. **tempo**: precisa ser mantida por 0,8 s para ser digitada.

Esses parâmetros ficam em `config.py`.

## Limitações e próximos passos
- **Letras com movimento (H, J, K, X, Z, Ç) não estão incluídas**: um classificador de frame único não vê trajetória. Um próximo passo seria guardar uma janela de ~30 frames e usar uma LSTM/GRU (PyTorch) ou características de trajetória da ponta do dedo.
- Algumas letras são parecidas por natureza (ex.: pares como U/V/R ou M/N). A matriz de confusão mostra quais são e ajuda a decidir onde coletar mais dados.
- O desempenho depende da diversidade do dataset: mais pessoas, ângulos e câmeras melhoram a generalização.
