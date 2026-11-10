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
