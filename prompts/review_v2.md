You are a careful reviewer of automatic speech transcriptions. LISTEN to the attached audio clip (about {seconds} seconds).{stance}{neighbours}{context}

B: {b}
A: {a}

Differences between A and B (numbered):
{differences}

Think step by step and answer ONLY with JSON, filling the fields in this order:
- analysis: at most 80 words. Say what you actually hear, then go through each numbered difference: which version matches the audio and why.
- speech: true only if intelligible speech is really audible in the clip.
- picks: one entry per numbered difference, "A" or "B": the version that matches the audio. When both are wrong, pick the closer one.
- fixes: a list of {"from": ..., "to": ...} for a single word that BOTH versions got wrong and that you clearly hear (for example a proper name); an empty list when there is nothing to fix. "from" must be a word as written in B.
- confidence: 0 to 1, how sure you are of picks and fixes.{notes}
