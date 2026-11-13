# Relatório consolidado da avaliação (2026-11-11)

Resumo de tudo o que foi medido, decidido e corrigido na bateria de avaliação, com a recomendação final. Os detalhes de cada
experimento estão em [modelos.md](modelos.md) (seções 6 e 7); o roteiro para repetir os testes, em [avaliacao.md](avaliacao.md);
o desenho da opção Qualidade, em [proposta-multiagente.md](proposta-multiagente.md).

## 1. Conclusões e recomendação

| Pergunta | Resposta |
|----------|----------|
| Qual o melhor motor **rápido**? | **Whisper `large-v3-turbo`** (OpenAI Whisper na XPU): WER 4,85 %, RTF 0,05 (~20x o tempo real). O Ollama `gemma4:e4b` tem a mesma velocidade, mas erra 7,87 % |
| O motor **Qualidade** (2 geradores + revisor) melhora o resultado? | **Sim, com a composição certa:** `gemma4:e4b` + Whisper `large-v3-turbo`, revisor `gemma4:12b`: **WER 3,99 %** (−0,86 ponto contra o melhor gerador sozinho, −18 % relativo; −3,9 pontos, −49 %, contra o `e4b` sozinho). Custa ~3x o tempo do Whisper sozinho (RTF 0,15, ~7x o tempo real) |
| O revisor importa? | **Muito.** Média dos 6 pares com Whisper: revisor `12b` 4,25 %, nemotron 5,23 %, `e4b` 5,53 %. Revisor pequeno **piora** o resultado (às vezes fica pior que o Whisper sozinho). O revisor tem que ser maior que os geradores |
| O nemotron-bolt é redundante com o `e4b`? | **Não são iguais: o `e4b` é melhor** (7,87 % contra 10,00 %; teste dos sinais p = 0,002). Como gerador não agrega; como revisor perde para o `12b`. Recomenda-se **escondê-lo das opções** |
| O `gemma4:12b` serve? | **Como transcritor não** (12 % de WER, ~12x mais lento: roda quase todo na CPU da Arc). **Como revisor, sim** (o melhor) |
| `lfm3-omni` / `lfm3-audio`? | **Removidos das opções** (WER de 22 a 69 % em português) |
| Vale incluir o Whisper nas opções? | **Sim.** É o ganho mais barato de qualidade: sozinho já vence o Ollama; combinado com revisor grande vence tudo |

**Configuração recomendada (já aplicada como padrão do motor Qualidade):** A = modelo de áudio do Ollama escolhido (`gemma4:e4b`),
B = OpenAI Whisper `large-v3-turbo`, revisor = `gemma4:12b`, limiar do portão 0,05, e em caso de dúvida vale a geração B (Whisper).
Se o `12b` não estiver instalado, o motor avisa e usa o modelo do A. Para uso do dia a dia sem revisão: Whisper `large-v3-turbo` sozinho.

## 2. Ambiente e material

- **Máquina:** Windows 11, GPU Intel Arc Pro B60 (23,8 GiB), Ollama 0.40.16 (Vulkan), PyTorch 2.15 com XPU, Python 3.13 em ambiente virtual `.venv` do projeto.
- **Áudio sintético:** 7 clipes (voz Microsoft Maria PT-BR e Zira EN; versões com ruído rosa a ~10 dB), referência conhecida.
- **Fala real:** conjunto público **FLEURS PT-BR** (Google, CC-BY 4.0): 6 clipes de ~4,6 min (132 falas, 2909 palavras), de vários falantes. Três clipes serviram para **ajustar** parâmetros (calibração) e três, com frases diferentes, para **validar** (frases novas).
- **Métrica:** WER com números por extenso (`25` = `vinte e cinco`); RTF = tempo de processamento ÷ duração do áudio (0,05 = 20x o tempo real).
- **Limites:** áudio lido (não espontâneo), um idioma, poucos clipes (uma diferença de ~1 ponto equivale a ~25 palavras), e só uma GPU. Os resultados dizem o que acontece **aqui**.

## 3. O que a medição revelou e foi corrigido

| Problema | Efeito | Correção |
|----------|--------|----------|
| "Traduzir" PT→PT pelo modelo | Dobrava o WER em fala real (~11 % → 5–7 %) e custava +45 % de tempo | Identificação de idioma (`langid.py`) pula a tradução se o texto já está no idioma de saída |
| `parallel` no Ollama | Sem ganho na Intel Arc (o "ganho de 35 %" inicial era carga do modelo na 1ª execução) | Registrado; segue configurável |
| Folga do VAD de 0,1 s | Cortava bordas de palavras | Folga de 0,3 s |
| Frases longas viravam "84 caracteres + resto" | Linhas acima de 42 caracteres | Divisão equilibrada que sempre cabe em 2 linhas |
| Executável de 901 MB | O PyInstaller varreu o Python global (PyTorch, `triton`…) | Build em ambiente limpo (63 MB) + exclusões |
| Console do Windows não imprimia acentos | Ferramentas quebravam | Saída em UTF-8 |
| `DiarizationPipeline(use_auth_token=…)` | WhisperX 3.8 mudou para `token=` | Aceita as duas; tenta os modelos pyannote em ordem |
| Whisper na XPU: `OUT_OF_RESOURCES` | Modelos do Ollama ocupavam a VRAM | A matriz descarrega os modelos do Ollama antes de cada geração Whisper |
| Processos de avaliação órfãos | Continuaram usando a GPU depois de pararem as tarefas | Encerrados à mão; lição: conferir processos ao parar execuções em segundo plano |
| Meu teste automático apagava `work/` | Quebrou uma avaliação em andamento | O teste usa pasta temporária |

## 4. Resultados principais

### 4.1 Geradores isolados (6 clipes de fala real)

| Gerador | WER total | Calibração | Frases novas | RTF |
|---------|-----------|-----------|--------------|-----|
| Whisper `large-v3-turbo` | **4,85 %** | 5,77 % | 3,88 % | 0,049 |
| Whisper `large-v3` | 4,95 % | 5,23 % | 4,65 % | 0,132 |
| Whisper `medium` | 6,50 % | 7,24 % | 5,71 % | 0,092 |
| `gemma4:e4b` | 7,87 % | 5,97 % | 9,87 % | 0,048 |
| nemotron-bolt `e4b` | 10,00 % | 6,57 % | 13,61 % | 0,076 |
| `gemma4:e2b` | 11,48 % | 12,27 % | 10,65 % | 0,041 |
| `gemma4:12b` | ~12 % | — | — | ~0,3 |

Os motores Whisper têm qualidade parecida entre as plataformas (OpenAI Whisper, faster-whisper, whisper.cpp, WhisperX): a escolha entre eles é de velocidade, dispositivo e recursos extras (WhisperX: palavra a palavra e diarização).

### 4.2 Motor Qualidade: pares de geradores × revisor (WER total, 6 clipes)

Prioridade em caso de dúvida: a geração G2 (Whisper, ou `e4b` sobre `e2b`). Limiar 0,05.

| G1 + G2 | Revisor `12b` | Revisor nemotron | Revisor `e4b` | Teto (revisor perfeito nos marcados) | RTF (`12b`) |
|---------|---------------|------------------|----------------|--------------------------------------|-------------|
| `e4b` + Whisper `turbo` | **3,99 %** | 4,50 % | 5,47 % | 3,44 % | 0,149 |
| `e4b` + Whisper `large-v3` | 4,16 % | 4,81 % | 5,60 % | 3,27 % | 0,234 |
| `e4b` + Whisper `medium` | 4,16 % | 6,50 % | 6,50 % | 4,06 % | 0,195 |
| `e2b` + Whisper `turbo` | 4,16 % | 4,81 % | 4,61 % | 3,58 % | 0,148 |
| `e2b` + Whisper `large-v3` | 4,26 % | 4,95 % | 4,61 % | 3,85 % | 0,232 |
| `e2b` + Whisper `medium` | 4,74 % | 5,78 % | 6,39 % | 4,37 % | 0,201 |
| `e2b` + `e4b` | 7,46 % | 7,87 % | 7,67 % | 6,63 % | — |

Referências: melhor gerador sozinho 4,85 % (Whisper `turbo`); `e4b` sozinho 7,87 %.
Leitura: (1) **par Ollama + Whisper vence qualquer par só de Ollama** (erros independentes: Whisper e Gemma erram em lugares diferentes; dois Gemma erram juntos);
(2) **revisor `12b`** leva todos os pares com Whisper para 4,0–4,7 %, perto do teto; os revisores menores perdem para o Whisper sozinho em vários pares;
(3) até um `e2b` pequeno, com Whisper e revisor `12b`, fica em 4,2 % (4,16 %): a qualidade vem do **Whisper + revisor grande**, não do tamanho do gerador Ollama;
(4) o Whisper `medium` (mais leve) com `e4b` e revisor `12b` já chega a 4,16 %, a mesma qualidade do `large-v3`.
O `e2b` + `e4b` com `12b` mostrou RTF de 1,2: é a 1ª configuração do revisor e inclui a carga do `12b`; não representa o regime normal (~0,15–0,23).

### 4.3 `gemma4:e4b` × nemotron-bolt `e4b` (comparação direta)

106 fragmentos: 65 saem idênticos; dos 41 diferentes, o `e4b` tem menos erros em 27 e o nemotron em 8 (teste dos sinais p = 0,002). WER entre os dois textos: 5,2 %.
Nemotron: 10,00 % contra 7,87 %, ~1,6x mais lento, e **derrubou o servidor do Ollama com chamadas em paralelo**. Como revisor, perde para o `12b`. **Recomendação:** esconder das opções (e, se quiser economizar disco, remover do Ollama: ~7,9 GB).

### 4.4 Outros resultados

- **Ollama (`gemma4:e4b`):** RTF ~0,023–0,05 (20–40x o tempo real); em fala real o WER caiu de ~11 % para 5–7 % depois da correção da tradução.
- **whisper.cpp:** um motor com backend automático; dois executáveis possíveis, CPU (9 MB) e CUDA (685 MB, só Windows x64); sem GPU NVIDIA o binário CUDA cai para a CPU. Velocidade com CUDA **não medida** (sem GPU NVIDIA). GPU Intel/AMD exige compilar com Vulkan.
- **Diarização (WhisperX):** funciona (diálogo de 3 turnos: trocas certas a ±1,8 s). Exige token do Hugging Face e a licença do modelo `pyannote/speaker-diarization-community-1` aceita.
- **Dispositivos:** CPU, CUDA/ROCm, Intel XPU e Metal escolhidos automaticamente; OpenAI Whisper e o alinhamento/diarização do WhisperX rodam na XPU; faster-whisper só em CPU/CUDA.
- **Google ADK + Ollama:** texto, chamada de ferramenta e ferramenta própria de áudio funcionam; áudio nativo via LiteLLM não (pela rota `/v1`, a chamada direta ao LiteLLM funciona, mas não dentro do ADK). Decisão do usuário: **sem LiteLLM**; agentes próprios sobre o `OllamaClient`.
- **Interface:** 14 verificações em um Edge real, sem erro de console. **Executável:** 63 MB, smoke test OK.

## 5. O que foi implementado nesta rodada

VAD com folga de 0,3 s; paralelismo e lote; `keep_alive` e pré-aquecimento; regras de legenda e quebra de linha; limpeza de alucinações; contexto/glossário; tempo por etapa; dispositivos; quatro motores Whisper (OpenAI Whisper, faster-whisper, WhisperX, whisper.cpp em duas builds com download automático); diarização; identificação de idioma que pula a tradução; **motor Qualidade** (pasta de fragmentos `work/<id>/`, portão, revisor, juiz, 2 ou 3 gerações); `.venv` isolado com todas as bibliotecas (`tools/setup_venv.py`); ferramentas de medição (`evaluate`, `quality_probe`, `quality_matrix`, `validate_engines`, sondas do ADK); catálogo enxuto (sem `lfm3`, sem Whisper antigo ou só inglês) e lista de modelos escondidos.

## 6. Limitações e riscos

- Fala lida e um idioma: **falta validar com fala espontânea, outros idiomas e áudio com música/ruído real**.
- A composição vencedora foi escolhida na matriz com 6 clipes (os 3 de calibração e os 3 de frases novas juntos): há risco de ajuste fino. Os ganhos foram consistentes nos dois conjuntos, mas valem uma validação com material novo.
- O revisor `12b` roda quase todo na CPU nesta Arc: em GPUs com mais VRAM o custo cai.
- O whisper.cpp CUDA e o desempenho em GPU NVIDIA não foram medidos.
- Os especialistas (glossário, pontuação, revisão da tradução) e a camada do ADK ainda **não** existem.

## 7. Pendências e decisões do usuário

1. Apagar do disco os modelos inúteis (`ollama rm`): `lfm3-omni`, `lfm3-audio` (~3,7 GB), `nemotron3:33b-q4_K_M` (27,6 GB, sem áudio) e, se concordar, o nemotron-bolt (~7,9 GB). Hoje estão só escondidos das opções.
2. Cancelar um trabalho apaga os fragmentos? (hoje ficam; limpeza automática após 7 dias.)
3. Trocar o token do Hugging Face (de escrita) por um de leitura.
4. Próximos passos técnicos: validação com fala espontânea; especialistas de glossário/pontuação; camada ADK; whisper.cpp com Vulkan para a Intel Arc.

## 8. Complemento: teste em áudio de filme (2026-11-11)

Detalhes em [experimento-longa-metragem.md](experimento-longa-metragem.md). Pipeline completo em 30 min do centro de um longa-metragem (saída pt-BR), contra a legenda original embutida:
**WhisperX** foi o mais fiel (WER do inglês 8,6 %, fidelidade 0,91–0,97), o **Whisper `turbo`** o mais rápido (2 min 7 s; 14x o tempo real) com significado equivalente, o **`gemma4:e4b`** ficou fraco
(WER 38,6 %, fidelidade 0,63) e o modo **Qualidade** (revisor `12b`) igualou o texto do Whisper sozinho a 4,7x o tempo (**não repetiu o ganho obtido em fala lida**). Conclusão: o ranking de geradores depende do tipo de áudio;
para filmes e áudio com música, usar WhisperX (e melhorar o VAD, testar o WhisperX como gerador B e reaproveitar os tempos reais do Whisper no modo Qualidade).


## 9. Complemento: laço de refinamento, especialistas e v2 (2026-11-11, noite)

Relatório completo: [experimento-loop-refinamento.md](experimento-loop-refinamento.md). Em resumo: o laço de refinamento melhora fala limpa (WER 5,46 → 4,94 % com 1 refinamento), mas a v1 ficou **2,3 a 3,2 vezes pior que o WhisperX** num áudio de filme (o gerador A inventava fala em música e o juiz tratava o silêncio como falha). A **v2.1** corrige isso (filme 11,6 % contra 11,5 % do WhisperX; clipes 4,86 %) e a melhor combinação medida, **WhisperX + `e4b` + revisor `12b`, 2 passadas**, é o **modo de qualidade 2** (4,11 % contra 5,41 %, nenhum clipe pior). Feedback ao gerador só com termos; triagem do documento e especialistas sem ganho mensurável.


## 10. Complemento: modo de qualidade 1 com porteiro de fala (2026-11-12)

Relatório completo: [experimento-modo1-v2.md](experimento-modo1-v2.md). O **modo 1** (só modelos do Ollama, sem WhisperX) agora é `e4b` (gerador) + `12b` (revisor) + **porteiro de fala** (Silero): no filme de 45 min o WER caiu de **38,5 % para 25,5 %** (inserções 16,6 → 3,7 %; tempo 22 → 18 min); em fala limpa fica igual (7,68 %, o porteiro não descarta nada). O WhisperX sozinho fica em 11,5 % no filme e 5,41 % nos clipes: o modo 1 **não** o alcança (as substituições, 14 %, vêm de gerador e revisor da mesma família). A votação entre 3 hipóteses e a 3ª passada foram medidas e **descartadas** (ganhos dentro do ruído, com custo).
