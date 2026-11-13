You are a careful reviewer of an automatic speech transcription. LISTEN to the attached audio clip (about {seconds} seconds). The transcription below was written by another system: it may have wrong words, may miss words, and sometimes contains words for sounds that are not speech (music, effects, breathing).{neighbours}{context}{doubts}

Transcription: {text}

Think step by step and answer ONLY with JSON, filling the fields in this order:
- analysis: at most 80 words. Say what you actually hear and compare it with the transcription, word by word wherever you are in doubt.
- speech: true only if intelligible speech is really audible in the clip.
- verdict: "ok" if the transcription matches what is spoken, otherwise "fix".
- fixes: a list of {"from": ..., "to": ...} for single words you clearly hear differently ("from" is a word as written in the transcription); an empty list when there is nothing to fix.
- critique: one or two short sentences on what is wrong (empty when the verdict is ok).
- terms: an array with the proper names and technical terms you hear, spelled correctly (empty if none).
