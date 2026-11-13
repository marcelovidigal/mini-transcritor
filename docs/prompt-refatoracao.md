# Prompt de refatoração: motores alternativos e otimização de velocidade

## Contexto
O Mini Transcritor hoje usa apenas modelos multimodais do Ollama. O objetivo desta rodada é preparar o app para motores alternativos e ganhar velocidade, sem interromper nenhuma transcrição em andamento. Durante a implementação não devem ser executados testes, medições ou downloads que usem a GPU. Os testes devem usar mocks.

## Decisões já tomadas
- A regra "somente Ollama" passa a valer só para o motor padrão. Os motores Whisper, faster-whisper e WhisperX são opcionais, e o usuário os escolhe.
- Os modelos desses motores só são baixados quando o usuário os seleciona.
- No catálogo, um modelo só aparece como "disponível" depois de baixado localmente. Enquanto não estiver baixado, o rótulo é "baixar ao selecionar".
- Segue valendo a regra de tags explícitas, sem `latest`. Para o Whisper, isso significa o nome completo do modelo (por exemplo `large-v3`) e a quantização quando houver.
- Não incluir a medição de desempenho agora. Ela será feita depois, quando a transcrição em andamento do usuário terminar.

## Bloco A — Motores e modelos
1. **Seleção de motor** na tela de início: Ollama (padrão), OpenAI Whisper, faster-whisper e WhisperX.
2. **Catálogo estático por motor**, em arquivo de texto, com todas as versões de modelo de cada família e preferência pelas variantes com timestamps. Marcar quais já estão baixadas.
3. **Download sob demanda** ao selecionar, com progresso e estado "baixando".
4. **Instalação opcional das dependências** pesadas (torch, faster-whisper, whisperx). O app base e o exe continuam leves. Se o motor não estiver instalado, mostrar um aviso claro e o comando de instalação.
5. **Diarização como opção na tela de início**, visível só nos motores que a suportam. Hoje isso significa apenas o WhisperX, porque o Whisper e o faster-whisper não têm diarização própria.
   - **Dependência:** a diarização do WhisperX usa o pyannote, que exige um token do Hugging Face e a aceitação da licença do modelo. Precisa de campo de configuração para o token.
6. **Alinhamento forçado:** avaliado depois, com o WhisperX (item 1 da lista de qualidade).

## Bloco B — Velocidade (vale para todos os motores)
1. **Pular silêncio** com VAD (Silero, com `silencedetect` do ffmpeg como alternativa), sem enviar trechos sem fala.
2. **Cortes nas pausas de fala**, usando os limites do VAD no lugar do tempo fixo. Os cortes devem respeitar o tamanho mínimo e máximo de trecho que o modelo aceita.
3. **Paralelismo:** 2–4 trechos ao mesmo tempo, configurável e limitado pela VRAM. Os resultados devem ser reordenados pelo índice do trecho.
4. **Tradução em lote:** janelas de várias falas por chamada, com contexto das vizinhas. Cada fala deve ser devolvida individualmente, sem perder a ordem.
5. **Modelo carregado:** `keep_alive` longo e pré-aquecimento ao iniciar.

## Bloco C — Qualidade (já aprovado, entra em seguida)
- Regras tipo Netflix: até 42 caracteres por linha, até 2 linhas, limite de caracteres por segundo, duração mínima e máxima, e quebras por sintaxe. Os valores do guia de português devem ser conferidos antes de fixar as constantes.

## Ordem de execução
1. Bloco B, itens 1 e 2: VAD e cortes por fala. É a base dos itens seguintes.
2. Bloco B, itens 3 e 4: paralelismo e tradução em lote.
3. Bloco B, item 5: manter o modelo carregado.
4. Bloco A: motores, catálogo e downloads, e por fim a diarização.
5. Bloco C: regras Netflix.
6. Após a transcrição do usuário terminar: medição, conjunto de referência e decisão sobre alinhamento forçado.

## Critérios de aceite
- Os testes existentes continuam passando.
- Cada novo comportamento tem teste com mocks.
- A documentação é atualizada: catálogo, arquitetura, tutorial e README.
- Nada é baixado nem executado na GPU durante o desenvolvimento.
