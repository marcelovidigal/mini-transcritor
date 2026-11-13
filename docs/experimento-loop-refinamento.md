# Experimento: laço de refinamento, especialistas e a v2 do motor Qualidade

> **Data:** 2026-11-11. **Dados brutos (só métricas):** [dados/experimento-loop-refinamento-2026-11-11.json](dados/experimento-loop-refinamento-2026-11-11.json).
> **Código:** `transcritor/quality_engine.py`, `transcritor/judge_v2.py`, `transcritor/specialists.py`, `prompts/`; **medição:** `tools/quality_loop_experiment.py`.
> **Atualização (2026-11-12):** o modo de qualidade 1 foi refeito e medido em [experimento-modo1-v2.md](experimento-modo1-v2.md) (porteiro de fala; votação descartada; gerador `e4b`). Este documento vale para o **modo 2** e para a v1/v2/v2.1; as seções sobre o autorrefinamento (modo 1) estão no novo.
> Plano que originou a v2: [plano-pipeline-v2.md](plano-pipeline-v2.md). Contexto: [proposta-multiagente.md](proposta-multiagente.md), [relatorio-avaliacao.md](relatorio-avaliacao.md).

## 1. Veredito

| Pergunta | Resposta |
|----------|----------|
| O laço de refinamento melhora a qualidade? | **Sim, nos áudios de fala limpa:** o WER médio caiu de 5,46 % (passada 1) para 4,94 % com 1 refinamento e 4,76 % com 2 (v1). **O 1º refinamento rende cerca de 75 % do ganho**; o 2º acrescenta ~0,2 ponto, dentro do ruído. |
| O pipeline v1 era seguro? | **Não.** No áudio de filme ele ficou **2,3 a 3,2 vezes pior que o WhisperX sozinho** (26,7–36,5 % contra 11,5 %), porque o gerador A inventava fala em música/efeitos e o juiz tratava o silêncio do WhisperX como "falta de texto". |
| A v2 corrige isso? | **Sim.** No filme a v2 volta a **11,6–11,7 %** (empata com o WhisperX; inserções de 16 % para 0,9 %). Nos clipes curtos a **v2.1 (prompt neutro quando o A é confiável) fecha em 4,86 %**, igual à v1 (4,94 %), com **~12 % menos tempo** e sem o pior caso da v1. |
| O pipeline ganha do WhisperX sozinho? | **Pouco e só em fala limpa:** −0,55 ponto de WER (~10 % relativo) nos clipes curtos; **zero no filme** (11,6 contra 11,5 %), onde o A é fraco demais para acrescentar e o teto de ganho por seleção entre A e B é ~11,3 %. |
| Revisor `e4b` ou `12b`? | O `12b` vale com A pequeno (v1: 3,93 contra 4,70 com A `e2b`); na v2.1 o resultado é **misto** (depende do A) e o `12b` custa **~2,4x o tempo de revisão** (650 s contra 270 s no filme). Sem vencedor claro. |
| O que o gerador A deve receber no refino? | **Só os termos.** Termos + pontos em disputa: 5,95 %; termos + crítica do revisor: 5,61 %; só termos: 5,06 % (clipes curtos, A `e2b`/`e4b`). Dar mais informação **reduz a independência** dos erros e piora. |
| Triagem do documento inteiro (só texto)? | **Sem ganho mensurável** no filme (11,6–11,7 %). |
| Especialistas (termos, pontuação, tradução)? | **Sem ganho comprovado:** WER inalterado; fidelidade pt-BR +0,01 a +0,02 (tolerante), **dentro do erro da amostra (±0,03)**. |

**Recomendação.** Adotar a **v2.1** como motor do **modo de qualidade 2** (WhisperX + `e4b` + revisor `12b`; a v1 fica só como linha de base reproduzível), com **1 refinamento** (a configuração medida em todos os conjuntos). O **modo 1** (sem WhisperX) é o autorrefinamento com porteiro de fala: ver [experimento-modo1-v2.md](experimento-modo1-v2.md). O ganho esperado é modesto (~10 % de WER relativo em fala limpa) e condicional; **3 passadas na v2.1 não melhoram (seção 6.3)**. Os especialistas devem continuar opcionais até ter uma medida que os enxergue. O maior ganho potencial restante não está no desenho do pipeline, e sim em **outro gerador A** (seção 8).

## 2. O que foi testado

1. **Laço de refinamento** (pedido: 2 repetições = 3 passadas): passada 1 = A e B geram, portão de divergência, revisor com áudio, juiz; passadas seguintes regeneram os fragmentos ainda em dúvida e revisam de novo.
2. **WhisperX como gerador B** (modelo `large-v3-turbo`, CPU nesta máquina, sem alinhamento: só o texto), em sessão com `initial_prompt` por fragmento.
3. **Gerador A** ∈ {`gemma4:e2b`, `gemma4:e4b`, `gemma4:12b`} (o `12b` só nos clipes curtos) × **revisor** ∈ {`gemma4:e4b`, `gemma4:12b`} = os dois **modos de qualidade** do app (`quality-e4b`, `quality-12b`).
4. **Especialistas** (termos, pontuação, revisão da tradução) antes de montar a legenda.
5. **v2** (plano em [plano-pipeline-v2.md](plano-pipeline-v2.md)): WhisperX como âncora; silêncio é informação; confiança do A medida no arquivo; decisão por ponto de divergência com texto montado pelo código; correções de palavra pelo áudio; revisor com raciocínio em etapas (CoT) e vizinhos; refino independente; triagem do documento (opcional).

## 3. Dados e métricas

| Conjunto | Conteúdo | Referência |
|----------|----------|------------|
| **Filme (principal)** | 45 min (1:45:00–2:30:00) de um longa em inglês, **áudio cortado de forma exata** (sem o deslocamento de ~7 s do `-c copy`), 199 fragmentos, 565 falas, 3182 palavras | legenda original (inglês) recortada e limpa; **pt-BR** = a mesma legenda traduzida pelo tradutor do pipeline (`e4b`) e por `12b` |
| **Clipes curtos** | 6 clipes de fala lida (FLEURS pt-BR), 15 a 20 fragmentos cada | transcrição de referência do corpus |

Métricas: **WER** (números por extenso), decomposto em substituições, omissões e **inserções** (`metrics.error_breakdown`); chrF e WER do pt-BR contra as duas referências traduzidas; **fidelidade** por juiz cego (`gemma4:12b`, 160 falas, janela temporal estrita ±0,4 s e tolerante ±4 s); tempo equivalente (soma dos tempos gravados de cada chamada, `parallel=1`).
Conteúdo do filme **não** está no repositório: só métricas (regra do projeto).

## 4. Resultados da v1 (pipeline original)

### 4.1 Clipes curtos (WER %, média de 6 clipes; "A só" = gerador sozinho; WhisperX sozinho = 5,41)

| A | Revisor | A só | Passada 1 | 2 passadas | 3 passadas |
|---|---------|------|-----------|-----------|-----------|
| e2b | 12b | 11,33 | 4,21 | 3,93 | **3,75** |
| e2b | e4b | 11,33 | 5,54 | 4,70 | 4,56 |
| e4b | 12b | 7,86 | 4,67 | 4,49 | 4,24 |
| e4b | e4b | 7,86 | 6,29 | 6,26 | 5,47 |
| 12b | 12b | 15,07 | 6,04 | 5,18 | 5,30 |
| 12b | e4b | 15,07 | 6,01 | 5,07 | 5,21 |
| **Média** | | | **5,46** | **4,94** | **4,76** |

### 4.2 Filme (WER do inglês falado; WhisperX sozinho = **11,5 %**: 4,3 % substituições + 6,2 % omissões + 0,9 % inserções)

| A | Revisor | A só | Passada 1 | Final (2 passadas) | Inserções (final) |
|---|---------|------|-----------|--------------------|-------------------|
| e2b | e4b | 42,9 | 31,6 | **30,7** | 16,4 % |
| e2b | 12b | 42,9 | 27,8 | **26,7** | 16,2 % |
| e4b | e4b | 46,3 | 38,6 | **36,5** | 21,0 % |

(3 passadas, e2b + e4b: 30,4 %. As combinações com A em `12b` foram canceladas por tempo: a geração do `12b` leva mais de 1 h 30 em 45 min de áudio.)

## 5. Diagnóstico: por que a v1 piora o que o WhisperX já faz bem

Análise por grupo de fragmentos (A `e2b` + revisor `12b`; só métricas):

| Grupo | Fragmentos | Erros do A | Erros do B (WhisperX) | **Erros do final v1** |
|-------|-----------:|-----------:|----------------------:|----------------------:|
| **Só o A escreveu texto** (B vazio) | 37 | 443 | 0 | **436** |
| Ambos, concordam (≤ 5 %) | 21 | 13 | 52 | 12 |
| Ambos, divergem (> 5 %) | 132 | 908 | 308 | **396** |
| Total | 199 | 1364 | 365 | **849** |

1. **Silêncio tratado como falha.** Nos 37 fragmentos em que o WhisperX calou, em 19 **não há fala** na legenda original (música, efeito) e **317 das 461 palavras** escritas pelo A caem em trecho sem fala. O juiz entendia "B vazio" como "falta o B" e ficava com o A: **436 dos 849 erros finais**, mais que todos os erros do WhisperX no arquivo.
2. **Juiz simétrico num caso assimétrico.** Onde os dois divergem, o A erra 3 vezes mais que o B (908 × 308), mas o revisor escolhe errado com frequência e **a confiança que ele declara é sempre ≥ 0,9** (limiares 0,7–0,9 não mudam nada): final com 396 erros, pior que o B sozinho (308).
3. **Teto baixo.** O melhor resultado possível escolhendo por fragmento entre A, B e a mescla é **11,3–11,7 %** ≈ o B sozinho: com um A fraco, **selecionar** não melhora; só *corrigir* o B ouvindo poderia.

**Simulação sobre o cache, sem GPU (filme, passada 1):** atual 27,8 % → *B vazio = silêncio* **15,3 %** → *+ nunca escolher A* 14,0 % → B sozinho 11,5 %. Nos clipes curtos a regra do silêncio não muda nada (lá o B nunca vem vazio), mas *nunca escolher A* piora o par bom (`e4b` + `12b`: 4,67 → 5,12): **uma regra fixa não serve aos dois tipos de áudio**. Sinais que separam os casos **sem referência**: razão de palavras A/B (clipes 0,98–1,06; filme 1,19–1,23) e fração de fragmentos "só A" (clipes 0 %; filme 19–22 %).

## 6. Resultados da v2 e da v2.1

| Versão | O que mudou | Clipes curtos (WER %) | Filme (WER %) |
|--------|-------------|----------------------|---------------|
| WhisperX sozinho | — | 5,41 | 11,5 |
| v1 (2 passadas) | — | 4,94 | 26,7–36,5 |
| **v2** | silêncio, confiança do A no arquivo, decisão por ponto, CoT; prompt do revisor **ancorado no B** | 5,30 | **11,6–11,7** |
| **v2.1** | v2 + prompt do revisor **neutro quando o A é confiável** | **4,86** | **11,6–11,7** (igual à v2: no filme o A não é confiável) |

- **Por que a v2 perdeu o ganho dos clipes curtos:** o prompt dizia ao revisor que "B é a referência e geralmente mais confiável", e ele escolheu **só B em 90 %** das revisões (alterou 22 de 320 fragmentos). Com o prompt neutro (v2.1) ele alterou **18 a 28 fragmentos por combinação** e o WER final foi de 5,30 para 4,86.
- **Perfil de risco por clipe** (média das 6 combinações): a v2.1 ficou no pior caso **+0,18 ponto** acima do WhisperX; a v1 chegou a **+0,9 ponto** nos clipes e **+25 a +37 pontos** no filme.
- **Revisor da v2 no filme:** 171 revisões por combinação, **0 a 1 fragmento alterado**; correções de palavra propostas em 7 de 851 revisões (0,8 %); escolheu o A em 21 (2,5 %); disse "sem fala" em 70 % dos trechos em que só o A escreveu (a legenda original tem fala em ~49 % deles).
- **Tempo:** filme v2 = 1289–1728 s equivalentes contra 2051–2374 s da v1; a geração do WhisperX por fragmento (882 s, CPU) domina; revisão 270 s (`e4b`) ou 650 s (`12b`).

### 6.1 Ablações

| Ablação | Resultado | Leitura |
|---------|-----------|---------|
| **Feedback ao gerador A no refino** (clipes, A `e2b`/`e4b`, 24 combinações) | termos **5,06**; termos + crítica 5,61; termos + pontos em disputa (com CoT) 5,95 | mais informação ⇒ erros menos independentes ⇒ pior |
| **Triagem do documento** (só texto, janelas de 60 fragmentos) | filme 11,6–11,7 % (igual à v2) | sem ganho; custa 31–66 s |
| **Pontos em disputa no filme** | idêntico à v2 | sem fragmentos a refinar quando o A não é confiável |
| **Especialistas** (termos + pontuação) | WER inalterado; mudam 22–35 fragmentos | o WER normalizado ignora pontuação/caixa |
| **Especialista de tradução** | mudou 14–35 traduções. Com revisor `e4b`: chrF −0,002 nas duas referências; com revisor `12b`: +0,005 contra a referência traduzida pelo `12b` e −0,001 a +0,002 contra a do `e4b` | o ganho só aparece contra a referência do mesmo modelo: **viés de estilo**, não melhora real |

### 6.2 Fidelidade pt-BR no filme (juiz cego, 160 falas; erro-padrão ≈ ±0,03)

| Sistema (A `e2b` + revisor `12b`) | Estrita (±0,4 s) | Tolerante (±4 s) |
|-----------------------------------|------------------|------------------|
| v1, legenda completa | 0,794 | 0,894 |
| v2, texto do WhisperX traduzido (sem especialistas) | 0,784 | 0,887 |
| v2, com termos e pontuação | 0,778 | 0,900 |
| v2, com todos os especialistas (**produto completo**) | 0,787 | 0,903 |

A fidelidade é **igual** entre v1 e v2 e entre com e sem especialistas (diferenças abaixo do erro da amostra). Atenção: esse juiz mede se cada fala **original** foi transmitida; **não vê texto inventado** (por isso a v1 não perde aqui, embora tenha 16 % de inserções). A vantagem da v2 está na **precisão** (WER/inserções), não na cobertura.

### 6.3 3 passadas na v2.1

Com `--loops 2` (3 passadas) a v2.1 não melhorou: média de 6 clipes **4,95 %** (A `e2b` e `e4b`) contra **4,86 %** com 1 refinamento, e o tempo cresce. O 2º refinamento não rende nos clipes; o modo 2 fica com 1 refinamento (2 passadas), como medido (4,11 % com `e4b` + revisor `12b`).

## 7. Limitações

- **Um filme** (inglês, com referência em inglês e saída traduzida): o diagnóstico vale para áudio com música/efeitos; precisa de outro gênero e idioma antes de generalizar.
- **Clipes de fala lida** (6): as diferenças entre versões (0,1 a 0,5 ponto) estão perto do ruído; a média das 36 combinações ajuda, mas os clipes compartilham gerador B e revisores.
- **A em `12b` não foi medido no filme** e a v1 do filme ficou incompleta (3 de 6 combinações), por tempo.
- **Fidelidade** com 160 falas: erro-padrão ~0,03; não detecta texto inventado.
- WhisperX no filme foi **por fragmento e na CPU** (882 s); o resultado de texto não difere do processamento do arquivo inteiro, mas o tempo sim (otimização: lotes de fragmentos).
- A confiança do A depende de **dois limiares** (razão A/B > 1,10; "só A" > 5 %) calibrados com os dois tipos de áudio disponíveis.

## 8. Próximos passos recomendados

1. **Trocar o gerador A** por um modelo mais forte em fala em inglês/ruidosa: o teto de seleção no filme é ~11,3 %, ou seja, **nenhum A que erra 43 % acrescenta algo**. Candidatos já registrados em [otimizacoes-futuras.md](otimizacoes-futuras.md) seção 12.
2. **Corrigir omissões do WhisperX** (6,2 % do erro no filme): é o maior erro restante e nenhum gerador testado o recupera; avaliar VAD neural e reaproveitar tempos do Whisper.
3. **Medida para os especialistas:** uma avaliação que enxergue pontuação, nomes e tradução (juiz por fala com o texto completo, não só o WER).
4. **Lotes de fragmentos no WhisperX** (tempo) e **A em `12b` no filme**, se houver tempo de GPU.
5. Validar a v2.1 com **outro filme** e com **fala espontânea**.

## 9. Como reproduzir

```
python tools/experimento_filme/cortar_trecho.py <video> 1:45:00 45 <pasta>          # áudio exato + legenda original limpa
python tools/experimento_filme/traduzir_referencia.py <pasta> gemma4:e4b ""          # referência pt-BR (e "_12b")
python tools/quality_loop_experiment.py <filme.json> --cache <c> --out <o> --judge v2 --loops 1 --translate --generators gemma4:e2b gemma4:e4b
python tools/quality_loop_experiment.py real.json holdout.json --cache <c> --out <o> --judge v2 --loops 1   # clipes curtos
python tools/experimento_filme/juiz_fidelidade.py <pasta> <saida1.srt> <saida2.srt> --janela 0.4           # fidelidade
```
`--judge v1|v2`, `--loops N` (refinamentos), `--feedback terms|points|critique`, `--triage`. O cache torna cada nova variante barata (só revisão e refino são refeitos).
