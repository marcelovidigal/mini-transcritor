# Experimento: legenda pt-BR de um longa-metragem (trecho de 30 min)

Avaliação do **pipeline completo** do app (extrair o áudio do vídeo → fragmentar → transcrever → traduzir → legenda pt-BR) em áudio de um filme,
comparando quatro sistemas contra a **legenda original embutida** no próprio arquivo. Data: 2026-11-11. O material foi usado só como teste, localmente e de forma
temporária; este documento traz **apenas métricas** (nenhum trecho de diálogo), e áudio, legendas e saídas ficam fora do projeto.

## 1. Resumo

| Pergunta | Resposta |
|----------|----------|
| Qual sistema foi o mais **fiel** ao que foi falado? | **WhisperX** (WER do inglês 8,6 %, F1 94 %): texto limpo e tempos de fala precisos (mediana 0,12 s) |
| Qual foi o mais **rápido**? | **Whisper `turbo`**: 2 min 7 s para 30 min de filme (14x o tempo real), com qualidade próxima à do WhisperX no significado |
| A **legenda em pt-BR** é boa? | **Sim, com Whisper `turbo` ou WhisperX**: fidelidade de significado 0,91 (0–1) com alinhamento estrito, 0,96 com tolerância de tempo; 84–86 % das falas fiéis, 2–4 % erradas |
| O modo **Qualidade** (e4b + Whisper + revisor `12b`) compensou? | **Não neste filme.** Texto equivalente ao do Whisper sozinho (fidelidade 0,94 contra 0,96 com tolerância de tempo), mas **4,7x mais lento** e com tempos grosseiros |
| E o `gemma4:e4b` sozinho? | **Fraco em áudio de filme**: WER 38,6 %, fidelidade 0,63. Música e efeitos atrapalham modelos de áudio generalistas |
| Diarização | Funcionou tecnicamente (12 falantes, 107 trocas); sem referência de falantes não há como medir a taxa de erro |

**Conclusão principal:** o ranking de geradores **depende do tipo de áudio**. Na fala lida (FLEURS) o Whisper `turbo` e a composição "Qualidade" lideravam;
em áudio de filme, com música e efeitos, o **WhisperX** (VAD próprio e alinhamento) é o melhor, e a composição Qualidade, calibrada em fala lida, não repetiu o ganho.

## 2. Material e método

- **Fonte:** longa-metragem em inglês (BluRay 1080p, 2 h 52 min, áudio E-AC-3 5.1 e legendas embutidas em inglês).
- **Trecho:** 30 min do **centro** do arquivo (1 h 11 min 17 s a 1 h 41 min 17 s), cortado com cópia de fluxos (sem recodificar), arquivo de ~830 MB. **Não se usou o filme inteiro.**
- **Processo avaliado:** o do app, de ponta a ponta, partindo do vídeo: o ffmpeg extrai o áudio (mono, 16 kHz), o VAD fragmenta (133 fragmentos), os motores transcrevem e o `gemma4:e4b` traduz para **pt-BR** (idioma de saída em todos os sistemas).
- **Sistemas** (um por vez, GPU livre, Intel Arc Pro B60):
  1. `gemma4:e4b` (Ollama);
  2. OpenAI Whisper `large-v3-turbo` (XPU);
  3. WhisperX `large-v3-turbo` com **diarização** (reconhecimento na CPU; alinhamento e diarização na XPU);
  4. **Qualidade**: A = `gemma4:e4b`, B = Whisper `turbo`, revisor `gemma4:12b`, limiar 0,05, dúvida → B (a composição recomendada).
- **Referência (o balizador):** a **legenda original** em inglês (a normal; a versão para surdos **não** foi usada), limpa de marcas e sons e **recortada à janela do trecho**: 422 legendas, 2382 palavras.
  Como o alvo é a legenda em **pt-BR** e a referência está em inglês, a comparação é feita de quatro formas:
  1. **Transcrição:** o texto **em inglês, antes da tradução**, contra a legenda original (WER, precisão, cobertura, tempo);
  2. **Tradução:** a nossa saída em pt-BR contra a **legenda original traduzida para pt-BR**, com o **mesmo tradutor e prompt do pipeline** (`e4b`), de modo que a variação do tradutor se cancela; e, como teste de sensibilidade, contra a versão traduzida pelo `gemma4:12b`;
  3. **Fidelidade de significado:** um juiz cego (`gemma4:12b`, só texto) compara 160 falas sorteadas da original em inglês com o pt-BR mostrado no mesmo instante (2 = mesmo sentido, 1 = parcial, 0 = errado ou ausente); sem saber qual sistema gerou a saída. Duas tolerâncias de alinhamento: estrita (±0,4 s) e tolerante (±4 s, que separa "o texto está certo" de "apareceu na hora certa");
  4. **Tempo e formato:** erro de início das falas, cobertura temporal, regras de legenda (42 caracteres, 2 linhas, 17 caracteres/s) e tempo de geração.
- **Ajustes de alinhamento:** o corte do vídeo começa no keyframe anterior, então o áudio fica **7,21 s** adiantado em relação à legenda (estimado pela mediana de 279 pares de falas do WhisperX); os tempos das saídas foram corrigidos por esse valor antes de comparar.

## 3. Resultados

### 3.1 Tempo de geração (30 min de filme = 1807 s de áudio)

| Sistema | Total | RTF | Vezes o tempo real | Transcrição | Tradução |
|---------|-------|-----|--------------------|-------------|----------|
| Whisper `turbo` | **2 min 7 s** | 0,070 | 14,3x | 66,8 s | 56,6 s |
| `gemma4:e4b` | 5 min 33 s | 0,184 | 5,4x | 285,1 s | 46,0 s |
| WhisperX + diarização | 7 min 12 s | 0,239 | 4,2x | 381,4 s | 48,9 s |
| Qualidade (revisor `12b`) | 9 min 53 s | 0,328 | 3,0x | 551,3 s | 39,1 s |

A extração do áudio leva ~1,7 s e o VAD ~0,2 s em todos. Na Qualidade, a transcrição se divide em geração A 292 s, geração B 69 s e revisão 190 s (118 de 133 fragmentos revisados).
O reconhecimento do WhisperX rodou na CPU (o CTranslate2 não usa GPU Intel): em GPU NVIDIA seria bem mais rápido.

### 3.2 Transcrição em inglês (antes de traduzir) contra a legenda original

| Sistema | WER | Precisão | Cobertura | F1 | Casou no tempo | Erro de início (mediana / p90) |
|---------|-----|----------|-----------|----|----------------|--------------------------------|
| WhisperX | **8,6 %** | **94,9 %** | 92,3 % | **93,6 %** | 86 % | **0,12 s** / 1,32 s |
| Qualidade | 18,4 % | 85,6 % | 91,5 % | 88,4 % | — (por fragmento) | — |
| Whisper `turbo` | 18,9 % | 84,8 % | **92,7 %** | 88,6 % | 90 % | 0,36 s / 2,37 s |
| `gemma4:e4b` | 38,6 % | 70,5 % | 78,5 % | 74,3 % | — (por fragmento) | — |

Precisão = fração do que o sistema escreveu que existe na referência (o resto são inserções, como alucinações e sons transcritos como fala); cobertura = fração do que a referência diz que o sistema capturou.
Os sistemas por fragmento (`e4b` e Qualidade) geram um texto por fragmento de ~12 s e distribuem o tempo pelo comprimento do texto: o erro de início não é comparável (mediana ~3 s por construção).
O Whisper `turbo` tem a mesma cobertura do WhisperX, mas **10 pontos a menos de precisão**: o nosso VAD por energia deixa passar trechos de música e ruído, e o modelo "transcreve" esses trechos (18 trechos foram descartados como provável alucinação).

### 3.3 Legenda final em pt-BR

| Sistema | vs. referência traduzida (`e4b`) WER / chrF | vs. referência traduzida (`12b`) WER / chrF | Fidelidade (juiz) estrita | Fidelidade (juiz) tolerante |
|---------|---------------------------------------------|---------------------------------------------|---------------------------|-----------------------------|
| WhisperX + diarização | 23,5 % / 0,865 | 31,6 % / 0,816 | **0,912** (fiel 86 %, errada 4 %) | **0,966** (fiel 95 %, errada 2 %) |
| Whisper `turbo` | 24,1 % / **0,886** | 31,5 % / **0,840** | 0,909 (fiel 84 %, errada 2 %) | 0,959 (fiel 93 %, errada 1 %) |
| Qualidade | 25,7 % / 0,877 | 32,3 % / 0,840 | 0,819 (fiel 72 %, errada 9 %) | 0,944 (fiel 91 %, errada 2 %) |
| `gemma4:e4b` | 48,6 % / 0,794 | 52,2 % / 0,771 | 0,631 (fiel 53 %, errada 27 %) | 0,778 (fiel 68 %, errada 12 %) |

- A **ordem dos sistemas não muda** com a referência traduzida pelo `e4b` ou pelo `12b`: o resultado é robusto ao tradutor. Duas traduções boas diferem naturalmente (WER de 24–32 % entre elas), por isso o WER de tradução só vale na comparação entre os sistemas; o chrF e o juiz são mais informativos.
- Os três sistemas baseados em Whisper têm **significado praticamente equivalente** na legenda final (0,94–0,97 com tolerância). O `e4b` perde 20 a 30 pontos.
- A diferença entre a nota estrita e a tolerante mostra o **custo da imprecisão de tempo**: Qualidade 0,819 → 0,944 e `e4b` 0,631 → 0,778, enquanto os sistemas com tempo real quase não mudam (0,91 → 0,96).
- Cobertura temporal (a legenda aparece quando há fala): precisão/IoU do WhisperX 0,84/0,72; `turbo` 0,60/0,56; Qualidade 0,51/0,48; `e4b` 0,51/0,47. As legendas por fragmento ficam mais tempo na tela do que a fala dura.

### 3.4 Formato e leitura (regras de legenda aplicadas à saída pt-BR)

| Sistema | Legendas | Duração mediana | Acima de 17 caracteres/s | Menores que 0,83 s |
|---------|----------|-----------------|---------------------------|--------------------|
| `gemma4:e4b` | 282 | 4,9 s | 21 (7 %) | 2 |
| Qualidade | 293 | 4,8 s | 6 (2 %) | 5 |
| Whisper `turbo` | 502 | 1,9 s | 127 (25 %) | 44 |
| WhisperX | 399 | 1,6 s | 210 (**53 %**) | 56 |

Nenhum sistema teve linha acima de 42 caracteres nem mais de 2 linhas. A **velocidade de leitura** é o ponto fraco dos sistemas com tempo real: o diálogo do filme é rápido e a tradução para o português é mais longa que o inglês, então metade das legendas do WhisperX passa de 17 caracteres por segundo. As legendas por fragmento passam menos porque ficam mais tempo na tela, o que reduz a velocidade de leitura à custa da precisão do tempo.

### 3.5 Diarização (WhisperX)

12 falantes distintos e 107 trocas de falante. Dois falantes concentram 53 % das trocas (29 e 28), e sete aparecem em 5 trocas ou menos. A legenda original não traz rótulos de falante, então **não há como calcular a taxa de erro de diarização**: o resultado é uma verificação de plausibilidade (as vozes principais foram separadas), e para medir de verdade seria preciso um trecho com falantes anotados.

### 3.6 Concordância entre os sistemas (WER entre os textos em inglês)

Whisper `turbo` × WhisperX 15,8 %; Qualidade × `turbo` 13,4 %; Qualidade × WhisperX 16,3 %; `e4b` × Qualidade 25,5 %; `e4b` × WhisperX 34,3 %; `e4b` × `turbo` 35,5 %.
Os três sistemas baseados em Whisper concordam entre si (13–16 %), enquanto o `e4b` diverge de todos (25–36 %): quando se compara um texto com os demais, o `e4b` é o outro lado da divergência.

## 4. Análise

1. **WhisperX foi o melhor em fidelidade e em tempo de fala.** O VAD do próprio WhisperX (pyannote) isola a fala da música e dos efeitos, e o alinhamento por palavra dá tempos precisos. Resultado: 95 % de precisão contra 85 % do `turbo`.
2. **O Whisper `turbo` é o melhor custo-benefício** (14x o tempo real, mesmo significado). O que o separa do WhisperX são as inserções fora da fala, que vêm do nosso VAD por energia em áudio de filme.
3. **O `gemma4:e4b` não serve como gerador principal para áudio de filme** (38,6 % de WER): seu desempenho em fala lida (5,9 %) não se transfere para áudio com música e efeitos.
4. **O modo Qualidade não superou o Whisper sozinho aqui.** O texto ficou equivalente (WER 18,4 % contra 18,9 %; fidelidade tolerante 0,944 contra 0,959), a um custo de 4,7x o tempo do `turbo`.
   Três razões: (a) o gerador A (`e4b`) é fraco neste áudio, então A e B discordam em 89 % dos fragmentos (118 de 133) e o revisor acaba arbitrando quase tudo (20 escolhas de A, 46 de B, 48 mesclas); (b) a calibração foi feita em fala lida, em que A e B erravam em lugares diferentes; (c) o pipeline distribui o tempo da legenda de forma proporcional ao texto do fragmento, perdendo os tempos reais do Whisper.
5. **A recomendação de geradores depende do tipo de áudio.** Para fala limpa, o Whisper `turbo` (ou a composição Qualidade); para áudio de filme, programas com música, ruído ou vários falantes, o **WhisperX**.

## 5. Recomendações

- **Para legendar filmes e vídeos com trilha sonora:** usar o **WhisperX** (com diarização se precisar identificar quem fala); com o `turbo` do OpenAI Whisper para rapidez.
- **VAD:** trocar o VAD por energia (que deixa passar música e ruído) pelo VAD do WhisperX/pyannote ou por um VAD neural (Silero) também nos caminhos do OpenAI Whisper e do Ollama: é o ajuste mais barato para elevar a precisão do `turbo`.
- **Modo Qualidade:** (1) testar o **WhisperX como gerador B** em vez do OpenAI Whisper; (2) quando o texto final de um fragmento vier da geração B, **reaproveitar os tempos reais do Whisper** em vez da distribuição proporcional; (3) repetir a matriz de geradores × revisores em áudio de filme antes de manter a composição padrão.
- **Velocidade de leitura:** como metade das legendas passa de 17 caracteres/s com tempo real, juntar legendas curtas vizinhas, alongar o tempo final e encurtar a tradução quando possível (hoje só se estende o fim em até 0,5 s).
- **Diarização:** validar em material com falantes anotados.

## 6. Limitações

- **Um trecho de um único filme em inglês:** os resultados mostram o que ocorre neste tipo de áudio, não uma regra geral.
- **A referência é uma legenda, não uma transcrição literal.** Ela omite falas repetidas e exclamações, o que reduz a precisão de **todos** os sistemas por igual; o que vale é a comparação entre eles. A referência traduzida herda os erros do tradutor, e por isso há duas versões (`e4b` e `12b`).
- **O juiz é um modelo de linguagem** (`gemma4:12b`, mesma família do tradutor): pode ter viés; a nota é uma medida de significado, não de redação. O juiz é cego em relação ao sistema.
- **O reconhecimento do WhisperX rodou na CPU:** o tempo dele não representa uma GPU com suporte.
- **Estimativa de deslocamento:** o ajuste de 7,21 s vem do próprio WhisperX; os pares de falas do `turbo` indicam 6,87 s (diferença de 0,34 s, típica da imprecisão de segmento do Whisper).

## 7. Erros do experimento e correções (registro)

1. Primeiro cortei o trecho do minuto 30 ao 60; você pediu o centro. Refeito com o trecho central.
2. A legenda de referência extraída ia até 5648 s (o corte `-t` não vale para legendas): 3x maior que o trecho. Os WER mostrados durante a execução (~70 %) eram falsos; a referência foi recortada e tudo foi recalculado a partir das legendas salvas.
3. O corte de vídeo `-c copy` deslocou o áudio em 7,21 s em relação à legenda; corrigido na comparação (sem isso o juiz via 60 % das falas "sem texto").
4. A primeira rodada de Qualidade usou o padrão do script de avaliação (revisor `e4b` e prioridade A) e não a composição recomendada. Refeita com o revisor `12b` e prioridade B; a primeira rodada (WER 31,2 %) foi descartada.

## 8. Como repetir

Os scripts estão em [tools/experimento_filme/](../tools/experimento_filme/) (paths relativos; os arquivos do filme ficam fora do projeto). Passos:
1. cortar o trecho (`ffmpeg -ss <início> -t 1800 -i filme.mkv -map 0:v:0 -map 0:a:0 -c copy segmento_30min.mkv`) e extrair a legenda embutida do mesmo trecho (`-map 0:s:0 referencia.srt`);
2. `prep_ref.py` (limpa a legenda) e `make_ref_ptbr.py` / `make_ref_ptbr_12b.py` (traduzem a referência);
3. rodar os sistemas com `tools/evaluate.py` (`--engine ollama|openai-whisper|whisperx|quality`, `--save-srt`, `--label`, `--diarize`);
4. `offset.py` (deslocamento), `analyze.py` (métricas; **recorte da referência à janela**), `judge.py` (`JUDGE_WINDOW=0.4` e `4.0`).

Dados: [dados/experimento-longa-metragem-2026-11-11.json](dados/experimento-longa-metragem-2026-11-11.json). Contexto: [relatorio-avaliacao.md](relatorio-avaliacao.md), [modelos.md](modelos.md), [proposta-multiagente.md](proposta-multiagente.md).
