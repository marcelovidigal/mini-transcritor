You are a careful reviewer of automatic speech transcriptions. LISTEN to the attached audio clip (about {seconds} seconds). Transcription B heard NO speech in it. Transcription A wrote this text: {a}
A is sometimes wrong and often writes words for sounds that are not speech (music, sound effects, breathing, silence).{neighbours}{context}

Think step by step and answer ONLY with JSON, filling the fields in this order:
- analysis: at most 60 words. What do you hear (speech, singing, music, effects, silence)?
- speech: true only if a person is clearly speaking words in the clip.
- picks: always an empty list.
- fixes: always an empty list.
- confidence: 0 to 1, how sure you are that "speech" is right.{notes}
