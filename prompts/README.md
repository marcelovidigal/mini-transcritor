# Prompts dos agentes

Todos os prompts dos modos de qualidade (`quality-1`, autorrefinamento, e `quality-2`, WhisperX + revisão) ficam aqui; os agentes os
leem desta pasta (`transcritor/prompts.py`). O modo 1 usa `review_self*.md`/`transcribe_self.md`; o modo 2, `review_v2*.md`/`transcribe_refine.md`.

Regras dos arquivos:

- Um arquivo `.md` por prompt, em inglês (os modelos respondem melhor). Campos entre chaves, como `{context}`, são trocados pelo código;
  campo que o código não informa fica como está.
- O **último** fim de linha de cada arquivo é ignorado. Espaços no começo ou no fim da linha **contam**: vários trechos começam com
  um espaço de propósito para emendar no texto anterior (por isso `context_note.md`, `feedback_*.md` e `review_*.md` começam com espaço).
- Os esquemas JSON das respostas (veredito, confiança, crítica, termos) ficam no código (`OllamaClient.review`), não aqui.
- `tests/test_prompts.py` compara a renderização com `tests/golden_prompts.json`. Ao mudar um prompt de propósito, rode
  `python tests/golden_prompts_gen.py` para congelar o novo texto e registre o motivo (de preferência com a medição que o justifica).
- Para experimentar sem mexer no projeto: `MT_PROMPTS_DIR=<outra pasta com os mesmos nomes>`.

| Arquivo | Agente | Quando |
|---------|--------|--------|
| `transcribe.md` | gerador A (Ollama, o modelo escolhido na tela) | 1ª geração e refinamentos |
| `context_note.md` | A, revisor, especialistas, tradução | quando o usuário informa contexto/glossário (`{verb}` diz o que fazer com ele) |
| `feedback_previous.md`, `feedback_critique.md`, `feedback_terms.md`, `feedback_closing.md` | gerador A | passadas 2 e 3: versão final anterior + crítica do revisor + termos ouvidos |
| `whisper_terms.md` | gerador B (WhisperX) | passadas 2 e 3: os termos que viram o começo do `initial_prompt` (depois entra a versão final anterior) |
| `review.md`, `review_notes.md`, `review_previous.md` | revisor (modelo de áudio, e4b ou 12b) | fragmentos em dúvida; `notes` pede crítica e termos; `previous` explica o candidato C |
| `review_v2.md`, `review_v2_silence.md`, `review_v2_notes.md`, `review_stance_anchor.md`, `review_stance_neutral.md`, `review_neighbours.md` | revisor do modo 2 (v2.1) | pontos de divergência entre A e B; `silence` quando o B calou; `stance` = âncora no B ou neutro (A confiável); `neighbours` = fragmentos vizinhos como contexto |
| `transcribe_refine.md`, `feedback_points.md` | gerador A do modo 2 | passada de refino com raciocínio (`analysis` antes da transcrição) |
| `triage.md` | triagem do documento (opcional, só texto) | antes da revisão com áudio; sem ganho medido |
| `review_self.md`, `review_self_doubts.md` | revisor do modo 1 | UMA transcrição; `doubts` só na votação (desligada) |
| `transcribe_self.md` | gerador do modo 1 | refaz o fragmento com a crítica do revisor |
| `transcribe_literal.md`, `transcribe_careful.md` | hipóteses da votação (desligada no app) | instruções diferentes para o mesmo modelo |
| `polish.md` | especialista de pontuação | depois do laço, antes de montar a legenda |
| `translation_review.md` | especialista de tradução | depois da tradução, antes de montar a legenda |
| `translate.md`, `translate_batch.md` | tradução | falas já faladas em outro idioma que o de saída |

As travas em código (o que o juiz e os especialistas aceitam ou descartam) **não** estão aqui: ver `transcritor/quality_engine.py` e
`transcritor/specialists.py`.
