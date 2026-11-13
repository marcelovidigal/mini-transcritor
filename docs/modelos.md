# Modelos: critérios, catálogo e resultados de teste

## 1. Regras da lista de modelos

1. **Só entra modelo que o Ollama declara com a capability `audio`** (`ollama show <modelo>`), idealmente confirmado com um teste real.
2. **Nome sempre com tag descritiva** (parâmetros e quantização), nunca `latest`: `gemma4:e4b`, `nemotron3:33b-q4_K_M`, `lfm3-omni:2b-q4_K_M`. A tag certa se descobre em `https://ollama.com/library/<modelo>/tags`, comparando o hash com o de `latest`.
3. **Entre variantes (quantizações) do mesmo modelo, baixa-se uma só**: a mais adequada ao uso. Para o `nemotron3` (33B), a que fica logo abaixo de 30 GB: `33b-q4_K_M` (28 GB). As demais (`nvfp4`/`mlx` 21 GB; `mxfp8` 35 GB; `q8` 36 GB; `bf16`/`mlx-bf16` 66 GB) não foram baixadas.
4. A lista é o arquivo [audio_models.txt](../audio_models.txt) (estático, editável sem recompilar); os instalados aparecem destacados.

## 2. Modelos considerados e decisão

| Modelo | Situação | Decisão |
|--------|----------|---------|
| `gemma4:e4b`, `e2b`, `12b` | no Ollama, com `audio` (confirmado no Ollama local) | **na lista** (família padrão) |
| `gemma4:26b`, `31b` | no Ollama, **sem áudio** (API responde 400 "Failed to load image or audio file") | fora |
| `nemotron-3.5-bolt:e4b-q4_K_M` | no Ollama, `audio` declarado e **confirmado em teste**; 7,9 GB (8B, 4B efetivos; pós-treinado a partir do `gemma4:e4b`, mesmo projetor de áudio de ~1 GB) | **na lista** (alternativa ao Gemma; empata com o `gemma4:e4b`, ver §4) |
| `nemotron3:33b-q4_K_M` (Nano Omni, 33B) | a página tem o selo `audio`, mas a tag baixada declara só `completion, vision, tools, thinking` e a API responde **400 "Failed to load image or audio file"**; no registro as tags GGUF (`q4_K_M`, `q8`, `bf16`) têm **uma única camada, sem projetor multimodal**; `nvfp4`/`mxfp8`/`mlx` são formatos para NVIDIA/Apple | **fora** (não processa áudio no Ollama) |
| `nemotron-3.5-bolt:74b-a9b-iq2` | tem projetor, 25 GB, quantização IQ2 (agressiva) | fora por ora (**não testado**) |
| `lfm3-omni:2b-q4_K_M`, `lfm3-audio:2b-q4_K_M` | no Ollama, `audio` confirmado, 2,0 e 1,7 GB | **na lista**, com ressalva (fracos em português) |
| "Nemotron-3.5" (30B) | não existe com esse nome (`nemotron-3.5-lightning` é 30B, mas só texto); o `nemotron-3.5-bolt` é descrito como "código, raciocínio e ferramentas", porém **aceita áudio** (ver acima) | o `bolt:e4b` entra; demais Nemotron (`3.5-lightning`, `3.5-super`, `3-super`, `3-ultra`, `3-nano`, `cascade-2`, `mini`, `nemotron`) são só texto/imagem: fora |
| Qwen2.5-Omni, SALMONN, VibeVoice, Voxtral (Small), Phi-4 Multimodal | **não existem na biblioteca oficial do Ollama** (consulta ao registro e à busca do site) | fora, por ora |
| Família **Qwen** da biblioteca oficial (`qwen3.5`, `qwen3.8`, `qwen3-vl`, `qwen3`, `qwen2.5vl`, `qwen3-coder`, `qwen3-next`, `qwq`) | os instalados (`qwen3.5:9b`, `qwen3.8:27b`, `qwen3-vl:32b`, `qwen3:32b`) declaram só `completion, vision, tools, thinking` (**sem áudio**); não existem no registro `qwen3-omni`, `qwen2.5-omni`, `qwen3.5-omni`, `qwen2-audio`, `qwen3-audio` nem `qwen3-asr` (verificado em 2026-11-08) | fora: nenhum Qwen do Ollama transcreve áudio |
| `rockn/Qwen2.5-Omni-7B-Q4_K_M` (comunidade) | só tem camadas de modelo/template/params, **sem projetor multimodal** → não processa áudio | fora |
| `embeddinggemma-2`, `d1-omni` | têm badge de áudio, mas são de embeddings/decisão | fora |

Análise dos modelos fora do Ollama (NeMo ASR, Whisper e derivados): [pre-analise-asr.md](pre-analise-asr.md).

O app trabalha só com o Ollama; modelos fora dele (Qwen2.5-Omni, SALMONN, VibeVoice, Voxtral, Phi-4-MM) exigiriam outro backend (transformers/vLLM), que está fora do escopo atual. Reavalie quando o Ollama passar a suportá-los.

## 3. Metodologia do teste

- Ferramenta: transcrição por trechos (mesmo pipeline do app: trechos de ~12 s, `temperature=0`, idioma original), comparada a um **texto de referência conhecido** com **WER** (taxa de erro de palavras; menor é melhor; pontuação e acentos ignorados).
- Cinco clipes sintéticos (voz do Windows/SAPI): `aula` (43 s, voz robótica antiga, EN), `reuniao` e `clima` (≈ 20 s, voz Zira, EN), `legendas` (≈ 20 s, voz Maria, **PT-BR**) e `frase` (7 s, voz robótica antiga, EN).
- **Limitações importantes:** poucas amostras, voz sintética (não é fala humana real, sem ruído, sotaques ou vários falantes), um único falante por clipe, só EN e PT-BR. Serve para comparar modelos entre si neste cenário, **não** para prever o desempenho em gravações reais. Teste com seus próprios áudios antes de decidir.

## 4. Resultados (WER %, menor é melhor)

| Modelo | aula (EN, robótica) | reunião (EN) | clima (EN) | **legendas (PT-BR)** | frase (EN, robótica) | **média** |
|--------|:--:|:--:|:--:|:--:|:--:|:--:|
| `gemma4:e4b` | 65 | 2 | 0 | **17** | 43 | **25,4** |
| `lfm3-audio:2b-q4_K_M` | **22** | 2 | 0 | 55 | 50 | 25,8 |
| `lfm3-omni:2b-q4_K_M` | 29 | 0 | 0 | 75 | 57 | 32,2 |
| `gemma4:e2b` | 86 | 0 | 0 | 26 | 86 | 39,6 |
| `gemma4:12b` | 86 | 2 | 0 | 19 | 186* | 58,5 |
| `nemotron-3.5-bolt:e4b-q4_K_M` | 68 | **0** | **0** | 19 | **36** | **24,6** |
| `nemotron3:33b-q4_K_M` | ✗ | ✗ | ✗ | ✗ | ✗ | **não processa áudio** (erro 400) |

✗ = o Ollama recusou o áudio em todos os trechos (modelo sem projetor de áudio).

\* O `gemma4:12b` às vezes responde "I'm sorry, I cannot provide a transcription…" em vez de transcrever, o que inflaciona o WER.

Leitura:
- Com voz limpa em inglês, todos erram 0–2 %. A diferença aparece na voz robótica e, principalmente, no **português**: os Gemma (e4b 17 %, 12b 19 %, e2b 26 %) vão bem; os `lfm3` (55 % e 75 %) são claramente fracos. **O padrão `gemma4:e4b` se sustenta** para uso em português.
- `lfm3-audio:2b` (1,7 GB) é uma boa opção **leve para inglês**.
- Tempo: cada clipe levou ~1–3 s por modelo nesta máquina (GPU local).
- Em um teste anterior e menor (um clipe), o `lfm3-audio` parecia muito superior; o conjunto maior mostrou que isso era específico do clipe em inglês com voz robótica. Moral: uma amostra só engana.
- **`nemotron-3.5-bolt:e4b-q4_K_M` (7,9 GB)** ficou **empatado com o `gemma4:e4b`** (média 24,6 % contra 25,4 %; pt-BR 19 % contra 17 %; melhor em `frase`, levemente pior em `aula`). Faz sentido: é um pós-treino do Gemma 4 E4B com o mesmo projetor de áudio. **Não é um ganho** de qualidade; entra como alternativa e para quem prefere o ajuste da NVIDIA. Diferença de 1–2 pontos de WER, com 5 clipes sintéticos, **não é significativa**.
- **`nemotron3:33b-q4_K_M` (28 GB, 33B)**: apesar do selo "audio" na biblioteca do Ollama, **não aceita áudio** (HTTP 400). Custou ~52 min de download para descobrir; confirmar antes de baixar: `ollama show` (capabilities) só funciona depois do download, mas o **manifesto do registro mostra se existe a camada `projector`** (sem ela, não há áudio/visão no Ollama). O mesmo vale para o `gemma4:26b`.
- Conclusão: entre os Nemotron do Ollama, **só o `bolt:e4b`** (e, em teoria, o `bolt:74b-a9b`, não testado) processa áudio. Nenhum supera o Gemma `e4b` nos nossos testes.

## 5. Motores alternativos: família Whisper

Catálogo em [whisper_models.txt](../whisper_models.txt) (editável, ao lado do app). Todos os modelos têm timestamps; o WhisperX também alinha por palavra e faz diarização.

| Modelo | Parâmetros | Observação |
|--------|-----------|------------|
| `large-v3-turbo` | 809 M | Recomendado: qualidade próxima do `large-v3`, bem mais rápido |
| `large-v3` | 1,55 B | Maior qualidade; ~10 GB de VRAM |
| `medium` / `small` / `base` / `tiny` | 769 M / 244 M / 74 M / 39 M | Compromissos de velocidade |

Ficam **fora** do catálogo, de propósito (decisão de 2026-11-11): as gerações anteriores (`large-v1`, `large-v2`) e os modelos só para inglês (sufixo `.en` e os destilados `distil-*`). No whisper.cpp há ainda as versões quantizadas (`q5_0`/`q5_1`) dos mesmos modelos.

**Ainda não medimos o WER desses motores** (a avaliação foi adiada até a GPU ficar livre); os números acima são características dos modelos, não resultados do nosso teste. Diarização: crie um token em huggingface.co/settings/tokens, aceite a licença de `pyannote/speaker-diarization-3.1` e informe o token em *Configurações*.

## 6. Bateria de avaliação de 2026-11-11 (Intel Arc Pro B60, Ollama 0.40.16, PyTorch 2.15+xpu)

**Material:** 7 clipes sintéticos (voz Microsoft Maria PT-BR e Zira EN; ~590 s no total) com texto de referência conhecido: `pt_long2` (179 s), `pt_long` (94 s), as duas versões com **ruído rosa a ~10 dB** (`pt_noise2`, `pt_noise`), `pt_a` (32 s, clipe difícil), `en_a` e `en_b`. **WER com números por extenso** (`25` = `vinte e cinco`, ver `metrics.spell_numbers`). Limitações: voz sintética e limpa (fala real é mais difícil), poucos clipes (cada diferença de ~1 pp equivale a poucas palavras) e um clipe (`pt_a`) instável: o mesmo modelo vai de 11 % a 21 % só mudando o tamanho do trecho ou o prompt. **Não medimos o erro de tempo das legendas** (não há SRT de referência).

### 6.1 Ollama (`gemma4:e4b`): velocidade e ajustes
| Configuração | Resultado |
|--------------|-----------|
| Velocidade | RTF **~0,05** (≈20x o tempo real) com o modelo já carregado; a 1ª execução paga a carga do modelo (RTF 0,7 num clipe de 19 s) |
| `parallel` 1 / 2 / 4 | **sem ganho**: 28,8 s / 27,3 s / 27,6 s para os 5 clipes PT (repetido: 28,8 / 25,4 / 28,0). Uma medição anterior sugeria 35 % de ganho; era artefato da carga do modelo na 1ª execução. Na Intel Arc o Ollama parece serializar as chamadas; em outras GPUs pode valer a pena |
| Tradução em lote (8) × fala a fala | JSON estruturado funciona (`gemma4:e4b`, `12b`, `e2b`); sem diferença mensurável de tempo ou WER nesta máquina |
| VAD × cortes por tempo | **sem VAD teve o menor WER** (0,4 % de média nos 4 clipes PT longos contra 1,8 % com VAD de folga 0,1 s) e é ~60 % mais lento nestes clipes (quase sem silêncio). Com folga de **0,3 s** nas bordas do VAD, o WER médio cai para 1,4 %; pausa mínima de 0,6 s piora (2,1 %) |
| Tamanho do trecho | 20 s foi o melhor com VAD (1,1 %); 8 s, 12 s ficaram em 1,4–1,8 % |
| Contexto/glossário | sem efeito claro (`pt_a` oscila entre 11 % e 21 % com qualquer mudança de prompt) |

Ação tomada: folga do VAD de 0,1 s para **0,3 s** (`audio.detect_speech`). Em áudio real, com silêncios longos, o VAD também pula trechos sem fala (ganho de velocidade que o áudio sintético não mostra): **falta medir com fala real**.

### 6.2 Motores Whisper (modelo `small`, mesmos 7 clipes)
| Motor (dispositivo) | RTF (modelo carregado) | WER `pt_long2` / `pt_noise2` / `pt_long` / `pt_noise` / `pt_a` / `en_a` / `en_b` |
|---------------------|------------------------|--------------------------------------------------|
| **Ollama `gemma4:e4b`** (GPU) | ~0,05 | 0,3 / 0,9 / 2,2 / 3,9 / 17,0 / 0 / 0 % |
| OpenAI Whisper (**XPU**) | ~0,05 | 1,2 / 1,5 / 2,8 / 2,2 / 28,3 / 0 / 0 % |
| faster-whisper (CPU, int8) | ~0,12 | 1,2 / 1,5 / 2,2 / 1,7 / 28,3 / 0 / 0 % |
| whisper.cpp v1.9.2 (CPU) | ~0,10 | 1,2 / 1,5 / 2,8 / 2,2 / 24,5 / 0 / 0 % |
| WhisperX (CPU, alinhamento por palavra) | ~0,22–0,28 | 1,2 / 1,5 / 2,2 / 1,1 / 22,6 / 0 / 4,8 % |

Leitura: **os quatro motores Whisper têm qualidade equivalente entre si** e parecida com a do Ollama neste material (no `pt_a`, difícil, o Ollama foi melhor). Em velocidade, o OpenAI Whisper na **XPU** empata com o Ollama; em CPU, o whisper.cpp é ~15 % mais rápido que o faster-whisper e o WhisperX é o mais lento (inclui o alinhamento). A vantagem dos motores Whisper é o **timestamp real** (por trecho; por palavra no WhisperX), não a precisão do texto. O WhisperX só foi testado na CPU (o PyTorch dele ficou isolado, 2.8.0 CPU) e **sem diarização** (exige token do Hugging Face).

### 6.3 Sonda "Qualidade" (duas gerações do Ollama + portão de divergência)
`gemma4:e4b` (A) contra um segundo modelo (B), 1103 palavras, 5 clipes PT, erros atribuídos por fragmento:
| B | WER A | WER B | revisor perfeito em todos os fragmentos | só nos divergentes (o que o portão permite) | fragmentos com erro em A / marcados / acertos |
|---|-------|-------|------------------------------------------|---------------------------------------------|-----------------------------------------------|
| `gemma4:e2b` | 2,2 % | 2,6 % | 1,6 % | 2,2 % | 10 / 2 / 2 |
| `nemotron-3.5-bolt:e4b-q4_K_M` | 2,2 % | 2,1 % | 1,8 % | 2,1 % | 10 / 1 / 1 |
| `lfm3-omni:2b-q4_K_M` | 2,3 % | 21,9 % | 1,9 % | 2,2 % | 11 / 9 / 6 |

Conclusões: (1) o baseline já erra pouco (~2 %), então o teto de ganho de um revisor perfeito é de **0,3–0,6 ponto percentual**; (2) o portão de divergência **perde a maioria dos erros** (cobertura de 10–20 % com `e2b` e `bolt`): os modelos erram juntos (ex.: a mesma alucinação "copyright" em trechos ruins); (3) `lfm3-omni` diverge muito, mas por ser fraco em PT-BR (22–68 % de WER), sem ganho; (4) `nemotron-bolt` é ~7x mais lento que o `e4b`; (5) `gemma4:12b` **não é viável** aqui: roda quase todo na CPU (190–435 s por clipe contra 2–8 s do `e4b`). **Neste material sintético, a ideia de duas gerações com portão não se justifica.** Falta testar com **fala real** (espontânea, sotaques, ruído), onde os modelos erram mais e talvez de forma menos correlacionada.

### 6.4 Google ADK + Ollama + áudio (`adk_probe.py`, ADK 1.19.0, `gemma4:e4b`)
T1 texto simples **OK** · T2 chamada de ferramenta **OK** · T3 ferramenta própria `ouvir_fragmento` (usa o `OllamaClient`) **OK**, o agente transcreveu o fragmento · T4 áudio nativo na mensagem **FALHOU** (`litellm`: *Invalid content item type: audio_url*). Ou seja: o caminho previsto para o revisor (ferramenta própria) funciona; o áudio direto via LiteLLM não.

### 6.5 Interface e executável
Interface em um Edge real (Playwright): 14/14 verificações, sem erro de console. **Build do executável:** a build normal saiu com **901 MB** porque o PyInstaller varreu o Python global (`triton` 340 MB, `cv2`, `pyarrow`, `scipy`, `transformers`, `pandas`…); num **ambiente virtual limpo** (só `requirements.txt` + PyInstaller) saiu com **63 MB** e o smoke test passou. Use sempre um venv limpo para gerar o release.

## 7. Matriz da opção Qualidade (2026-11-11): geradores × revisores em fala real

Resumo; versão completa e recomendação em [relatorio-avaliacao.md](relatorio-avaliacao.md); dados brutos em `docs/dados/matriz-qualidade-2026-11-11.json`
(gerado por `tools/quality_matrix.py`). Material: 6 clipes de fala real FLEURS PT-BR (3 de calibração + 3 de frases novas; 2909 palavras). Regras: só **2 geradores** por
pipeline; o **Whisper só gera** (nunca revisa); o **revisor** é um modelo de áudio do Ollama; sem decisão do revisor vale a geração G2; limiar do portão 0,05.

**Geradores isolados (WER total):** Whisper `turbo` 4,85 % · Whisper `large-v3` 4,95 % · Whisper `medium` 6,50 % · `gemma4:e4b` 7,87 % · nemotron-bolt 10,00 % · `gemma4:e2b` 11,48 %.

| G1 + G2 | Revisor `12b` | Revisor nemotron | Revisor `e4b` | Teto (revisor perfeito) |
|---------|---------------|------------------|----------------|--------------------------|
| `e4b` + Whisper `turbo` | **3,99 %** | 4,50 % | 5,47 % | 3,44 % |
| `e4b` + Whisper `large-v3` | 4,16 % | 4,81 % | 5,60 % | 3,27 % |
| `e4b` + Whisper `medium` | 4,16 % | 6,50 % | 6,50 % | 4,06 % |
| `e2b` + Whisper `turbo` | 4,16 % | 4,81 % | 4,61 % | 3,58 % |
| `e2b` + Whisper `large-v3` | 4,26 % | 4,95 % | 4,61 % | 3,85 % |
| `e2b` + Whisper `medium` | 4,74 % | 5,78 % | 6,39 % | 4,37 % |
| `e2b` + `e4b` | 7,46 % | 7,87 % | 7,67 % | 6,63 % |

**Conclusões:** (1) Ollama + Whisper vence qualquer par só de Ollama (erros independentes); (2) o revisor tem de ser **grande**: o `12b` (ruim como transcritor, 12 %) é o melhor revisor e leva todos
os pares com Whisper a 4,0–4,7 %, enquanto o `e4b` e o nemotron como revisores ficam, em vários pares, **piores que o Whisper sozinho**; (3) melhor combinação: `e4b` + Whisper `turbo` + revisor `12b`, WER 3,99 %
(−0,86 ponto, −18 %, contra o Whisper sozinho; −49 % contra o `e4b` sozinho), RTF ~0,15; (4) **`e4b` × nemotron-bolt:** não são iguais, o `e4b` é melhor (7,87 % × 10,00 %; teste dos sinais p = 0,002),
logo o nemotron não agrega como gerador e perde para o `12b` como revisor; (5) `lfm3-omni` e `lfm3-audio` saíram das opções (WER de 22 a 69 % em português).
