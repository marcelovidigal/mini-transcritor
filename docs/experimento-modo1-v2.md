# Experimento: o modo de qualidade 1 (autorrefinamento), o porteiro de fala e a votação

> **Data:** 2026-11-12. **Dados brutos (só métricas):** [dados/experimento-modo1-2026-11-12.json](dados/experimento-modo1-2026-11-12.json).
> **Código:** `transcritor/speechgate.py` (porteiro), `transcritor/vote.py` (votação), `transcritor/quality_engine.py` (`_after_generation_self`, `_vote`),
> `prompts/` (`review_self*.md`, `transcribe_literal.md`, `transcribe_careful.md`); **medição:** `tools/quality_loop_experiment.py` (`--pairs`, `--gate`, `--vote`).
> Continua o [experimento-loop-refinamento.md](experimento-loop-refinamento.md) (que mediu o autorrefinamento antes do porteiro e definiu o modo 2).

## 1. Veredito

| Pergunta | Resposta |
|----------|----------|
| Qual é o modo 1 do app agora? | **`gemma4:e4b` gera, `gemma4:12b` revisa, porteiro de fala (Silero) antes do gerador, 2 passadas, sem votação, sem especialistas.** Modelos fixos. |
| Quanto melhorou? | No filme de 45 min: **38,5 % → 25,5 % de WER** (−13,0 pontos) e **22,1 → 17,8 min** de tempo equivalente. Em fala limpa o resultado não muda (7,68 %), porque o porteiro não descarta nada. |
| Chegou perto do WhisperX? | **Não.** WhisperX sozinho: 11,5 % no filme e 5,41 % nos clipes. O modo 1 novo fica **14 pontos acima no filme** e **2,3 pontos acima nos clipes**. As substituições (14,2 %) são 3,3 vezes as do WhisperX (4,3 %). É a opção **sem WhisperX/PyTorch**, não uma substituta do modo 2. |
| De onde veio o ganho? | **Quase todo do porteiro de fala.** Ele corta o texto inventado em música e efeitos (inserções de 16,6 % para 3,7 %) e, de quebra, poupa tempo, porque 46 de 199 fragmentos nem chegam aos modelos. |
| A votação entre 3 hipóteses ajuda? | **Não.** No filme (sobre o porteiro): com `e4b` melhora 0,3 ponto, com `e2b` **piora** 0,9, e custa 5 minutos extras. Nos clipes: −0,76 ponto com `e2b` (IC95 −1,82 a +0,39, melhora em 5 de 6) e −0,13 com `e4b` (IC −1,58 a +0,92, melhora em 2 de 6): **dentro do ruído**. A votação foi **descartada** (o código fica, desligado: `quality_vote`). |
| A 3ª passada compensa? | **Não.** `e2b → 12b`: 33,1 → 32,7 % (filme) e 10,75 → 10,59 % (clipes); `e4b → 12b`: 38,5 → 35,0 % (filme, só inserções), 7,68 → 7,79 % (clipes, piora). Fica em 2 passadas. |
| O que o laço não corrige? | **Substituições** (14–18 %): o revisor da mesma família de modelos ouve os mesmos erros do gerador. Só um gerador independente (o WhisperX, modo 2) resolve. |

## 2. O que foi testado

1. **Autorrefinamento sem WhisperX** (já medido no experimento anterior): o gerador transcreve; o revisor, com o áudio, julga UMA transcrição (raciocínio em etapas, veredito, correções de palavra, crítica, termos; "sem fala" vira silêncio); o que não está ok volta ao gerador com a crítica; trava contra reescrita excessiva (WER ≤ 0,35 contra a tentativa anterior). Cinco pares gerador → revisor, 2 e 3 passadas.
2. **Porteiro de fala** (`speechgate.py`): antes do gerador, o **Silero VAD** (ONNX, vem com o `faster-whisper`; sem PyTorch) mede a fração de fala de cada fragmento; com **≤ 3 %** o fragmento é descartado (nem gerador, nem revisor). Sem o Silero instalado, ou se ele não achar fala em fragmento nenhum, o porteiro se desliga e tudo é transcrito (com aviso).
3. **Votação** (`vote.py`): o gerador transcreve cada fragmento 3 vezes (instrução padrão; literal; "atenta" com áudio normalizado em ganho); o código alinha as três palavra a palavra, aplica as trocas que 2 de 3 repetem (consenso) e transforma as discordâncias em **perguntas objetivas** ao revisor ("atual × outra"); o texto final é montado pelo código a partir das escolhas; se as 3 hipóteses concordam pouco (< 0,5) o revisor é avisado de que provavelmente não há fala clara.
4. **Configurações medidas** (revisor `12b`, 2 passadas): porteiro + votação, só votação, só porteiro (todas com gerador `e2b`) e porteiro + votação com `e4b`; depois, só porteiro com `e4b` (controle, a configuração final).

## 3. Dados e métricas

Os mesmos do experimento anterior: **filme** de 45 min (199 fragmentos, referência = legenda original recortada) e **6 clipes de fala lida** (FLEURS pt-BR, 15 a 20 fragmentos cada). Métrica: **WER** do texto falado (números por extenso), decomposto em substituições/omissões/inserções (`metrics.error_breakdown`); **tempo equivalente** = soma dos tempos registrados no cache de chamadas (reexecutar é barato). Toda comparação usa **duas referências**: o **gerador sozinho** (`e2b` 42,9 % / `e4b` 46,3 % no filme; 11,33 / 7,86 % nos clipes) e o **WhisperX sozinho** (11,5 % no filme; 5,41 % nos clipes). Conteúdo do filme **não** está no repositório: só métricas. Os clipes aparecem anonimizados nos dados exportados.

## 4. Resultados

### 4.1 Calibração do porteiro (por que o Silero)

Sinais simples de sinal (nível, fração de quadros ativos, energia na faixa da voz, modulação silábica 2–8 Hz, fluxo espectral) **não separam** a música/efeitos do filme da fala: os trechos sem fala são tão altos e contínuos quanto os de fala e as faixas dos cinco sinais se sobrepõem. O **Silero** separa. Fragmentos sem fala = 35 (menos de 1 palavra na referência); fala = os demais.

| Limiar do Silero | Fragmentos descartados | Corretos (sem fala) | Palavras de fala perdidas |
|---|---|---|---|
| 0,50 | 52 | 34 | 89 (2,8 % da referência) |
| **0,35 (escolhido)** | **46** | **31** | **59 (1,9 %)** |
| 0,20 | 33 | 22 | 39 (1,2 %) |

O limiar do meio foi fixado **antes** de ver os resultados dos modos. O custo são trechos de fala sob música alta, descartados por engano (as omissões sobem de ~6 % para ~8 %). Nos 6 clipes o porteiro descartou **0** fragmentos e os resultados são idênticos aos de sem porteiro.

### 4.2 Filme de 45 min (2 passadas, WER %; subst./omissões/inserções)

| Configuração | WER | Subst. | Omiss. | Inserç. | Tempo equiv. |
|---|---|---|---|---|---|
| **WhisperX sozinho (referência)** | **11,5** | 4,3 | 6,2 | 0,9 | — |
| `e2b` sozinho (referência) | 42,9 | 18,8 | 5,7 | 18,5 | — |
| `e4b` sozinho (referência) | 46,3 | 16,9 | 6,4 | 22,9 | — |
| `e2b → 12b` (modo 1 antigo) | 33,1 | 17,3 | 5,9 | 10,0 | 22,7 min |
| `e2b → 12b`, só votação | 31,8 | 17,6 | 5,6 | 8,6 | 28,1 min |
| `e2b → 12b`, só porteiro | 27,6 | 15,2 | 8,2 | 4,2 | 18,1 min |
| `e2b → 12b`, porteiro + votação | 28,5 | 15,6 | 7,9 | 5,0 | 23,0 min |
| `e4b → 12b` (modo 1 antigo com `e4b`) | 38,5 | 15,6 | 6,3 | 16,6 | 22,1 min |
| `e4b → 12b`, porteiro + votação | 25,2 | 14,0 | 7,6 | 3,6 | 23,1 min |
| **`e4b → 12b`, só porteiro (modo 1 atual)** | **25,5** | 14,2 | 7,6 | 3,7 | **17,8 min** |

Leituras:
- O **porteiro sozinho** reduz o `e2b → 12b` em 5,5 pontos e o `e4b → 12b` em 13,0, e deixa ambos mais rápidos.
- O gerador sozinho com porteiro já cai de 42,9/46,3 % para **28,9/28,3 %**: a diferença entre o `e2b` e o `e4b` quase some quando o ruído sai, e o revisor `12b` tira mais 1,3 a 2,8 pontos (substituições).
- A **votação** não acrescenta: o consenso sozinho (sem revisor) ficou em 29,2 % contra 28,9 % do gerador (`e2b`). Aplicou 24–37 trocas por maioria e levantou 201–313 dúvidas, sem ganho.
- O erro restante do modo 1 é de **substituições** (14,2 %, 3,3× o WhisperX) e **omissões** (7,6 %; 1,4 ponto acima do WhisperX, em parte por causa do porteiro).

### 4.3 Clipes curtos (6 clipes, WER médio %, 2 passadas)

| Configuração | `e2b → 12b` | `e4b → 12b` |
|---|---|---|
| WhisperX sozinho (referência) | 5,41 | 5,41 |
| Gerador sozinho (referência) | 11,33 | 7,86 |
| Autorrefinamento antes (sem porteiro) | 10,75 | 7,68 |
| 3 passadas | 10,59 | 7,79 |
| Porteiro (descartou 0 fragmentos) | 10,75 | 7,68 |
| Porteiro + votação | 9,99 | 7,55 |

Votação contra o autorrefinamento (diferença por clipe, 6 clipes): `e2b` −0,76 ponto (IC95 de bootstrap −1,82 a +0,39; melhora em 5 de 6; por clipe +1,82, −0,54, −0,47, −2,71, −0,78, −1,86); `e4b` −0,13 (IC −1,58 a +0,92; melhora em 2, igual em 1, piora em 3). **Nenhum dos dois é distinguível de zero.**

### 4.4 Autorrefinamento sem porteiro: os cinco pares (resultado do experimento anterior, 2 passadas)

| Par (gerador → revisor) | Filme (WhisperX 11,5) | Clipes (WhisperX 5,41) |
|---|---|---|
| `e2b → 12b` | 33,1 (3ª passada: 32,7) | 10,75 |
| `e2b → e4b` | 34,5 | 11,79 |
| `e4b → 12b` | 38,5 (3ª passada: 35,0) | 7,68 |
| `e2b → e2b` | 40,3 | 11,10 |
| `e4b → e4b` | 43,0 | 8,18 |

## 5. Diagnóstico

1. **A maior parte do erro do modo 1 antigo era texto inventado.** Em música/efeitos o gerador pequeno escreve palavras (inserções de 18,5–22,9 %). O revisor cortava só parte delas (para 10–17 %), porque só vê o fragmento depois de o gerador escrever e, às vezes, aceita o texto. O porteiro tira o problema da entrada, antes do gerador, com um detector que já sabe o que é voz.
2. **Por que o `e4b` rende mais que o `e2b` agora.** Com o ruído fora, o `e4b` é o gerador mais forte (nos clipes, 7,86 % contra 11,33 %) e o `12b` o revisa melhor: 28,3 → 25,5 % (`e4b`) contra 28,9 → 27,6 % (`e2b`). Sem porteiro era o contrário, porque o `e4b` inventava mais (22,9 % de inserções).
3. **Por que a votação não funcionou.** As 3 hipóteses saem do **mesmo modelo**, com erros correlacionados: onde ele se engana, se engana nas três; onde discorda, são variações de ruído, sem sinal sobre qual está certa. Das 245–313 dúvidas por filme, o revisor não as converteu em ganho. O consenso por maioria também não corrige o que é sistemático.
4. **Piso do desenho sem WhisperX.** As substituições ficam em 14–18 % porque gerador e revisor ouvem o mesmo áudio com a mesma família de modelo. Para passar disso é preciso um gerador independente (modo 2).

## 6. Defeitos encontrados (e corrigidos) no caminho

- **Hipóteses da votação sem limite de tokens:** em trecho sem fala o `e2b` repete texto até estourar o tempo (300 s por chamada); uma rodada "só votação" ficou 95 min parada. Corrigido com `MAX_VARIANT_TOKENS = 512` (mesmo padrão dos agentes de revisão, `MAX_JSON_TOKENS`); teste adicionado. Na primeira rodada (porteiro + votação, `e2b`) 3 hipóteses falharam por tempo e ficaram de fora do consenso (3 de 304).
- **Porteiro que descarta tudo:** num áudio em que o Silero não acha fala em fragmento nenhum, o trabalho falharia com "o modelo não retornou texto". Agora o porteiro se desliga nesse caso (com aviso) e transcreve tudo.
- **Comparação justa:** a linha de base do autorrefinamento antigo deve ser a de **2 passadas** (o arquivo da 3ª passada reporta o `final` da 3ª). Todas as tabelas acima usam 2 passadas.

## 7. Limitações

- **Um filme** (inglês, com música e efeitos) e **6 clipes de fala lida**. O ganho do porteiro é o resultado de uma amostra e o limiar foi calibrado **no mesmo filme** (três valores vistos, o do meio foi fixado antes dos resultados dos modos): pode estar otimista em outro filme. Os clipes só confirmam que o porteiro não prejudica fala limpa.
- A votação tem barras de erro que incluem zero nos clipes; não está provado que **não** ajuda (poderia ajudar com outro gerador ou outras variações), só que **não compensa o custo** nas condições medidas.
- Idioma: filme em inglês, clipes em pt-BR; o Silero é multilíngue, mas não medimos outros idiomas.
- O porteiro descarta falas sob música alta (≈ 2 % das palavras de referência), um custo aceito conscientemente; um limiar menor (0,20) perderia 1,2 %, mas deixaria passar 13 fragmentos sem fala em vez de 4.

## 8. Próximos passos recomendados

1. **Validar o porteiro em outros filmes/idiomas** e, se necessário, calibrar o limiar por uma medida sem referência (por exemplo, a fração de fragmentos descartados).
2. **Omissões e substituições:** testar um gerador independente leve (Whisper pequeno) no modo 1 só quando o usuário aceitar a dependência, ou os modelos de [otimizacoes-futuras.md](otimizacoes-futuras.md) seção 12.
3. **Revisão mais específica:** pedir ao revisor só os trechos com palavras de baixa probabilidade (se o modelo expuser logprobs) em vez de revisar todos.
4. Medir o **`e4b` com porteiro nos 6 clipes** só para completar a tabela (o resultado é idêntico por construção, pois o porteiro descarta 0).

## 9. Como reproduzir

```
python tools/quality_loop_experiment.py <filme.json> --cache <c> --out <o> --loops 1 --pairs gemma4:e4b/gemma4:12b --gate     # modo 1 atual
python tools/quality_loop_experiment.py <filme.json> --cache <c> --out <o> --loops 1 --pairs gemma4:e2b/gemma4:12b --gate --vote
python tools/quality_loop_experiment.py real.json holdout.json --cache <c> --out <o> --loops 1 --pairs gemma4:e4b/gemma4:12b   # clipes
```
`--gate` liga o porteiro, `--vote` a votação, `--loops N` são os refinamentos (1 = 2 passadas). O cache torna cada variante barata (só o que mudou é refeito).
