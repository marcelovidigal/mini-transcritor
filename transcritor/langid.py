"""Identificação leve de idioma por palavras funcionais e alfabeto (sem dependências).

Serve para o pipeline do Ollama saber se o texto transcrito JÁ está no idioma de saída: nesse caso a "tradução" é
pulada. (Medido em 2026-11-11 com fala real em PT-BR: passar o texto pelo modelo para "traduzir" português para
português dobrou o WER, de ~6 % para ~11 %, e custou ~40 % mais tempo.)
"""
import re

_WORDS = {
    "pt": "de que o a e do da em um uma para com não nao os as no na por mais se dos das como mas foi ao ele ela seu sua ou "
          "quando muito já também é são está isso esse essa entre depois sobre ser tem há pelo pela nos eu você vamos hoje "
          "então ainda cada sem pode podem foram será seria onde qual quais nosso nossa deste desta nesta neste aos "
          "ninguém alguém também porém apenas mesmo mesma tudo nada algo sempre nunca agora aqui ali lá vocês nós eles elas "
          "meu minha dele dela muita muitos pouco pouca todos todas outros outras enquanto pois assim só até lembre dia",
    "es": "de la que el en y a los del se las por un para con no una su al lo como más pero sus le ya o este sí porque esta "
          "entre cuando muy sin sobre también me hasta hay donde quien desde todo nos durante son está fue ser tiene "
          "puede pueden cada otro otra estos estas eso",
    "en": "the of and to in is that it for was on are as with his they at be this from have or by one had not but what all "
          "were we when your can said there use an each which she do how their if will about out up them then so some her "
          "would make like him into time has look two more write go see number no way could people my than first water been",
    "fr": "le la les de des du et en un une que qui dans est pour pas sur au avec ce il elle ne se plus par mais ou son sa "
          "ses nous vous comme tout être fait aux cette sont ont était leur où très aussi",
    "de": "der die das und in den von zu mit sich des auf für ist im dem nicht ein eine als auch es an werden aus er hat "
          "dass sie nach wird bei einer um am sind noch wie nur über wurde oder aber vor zur kann",
    "it": "di che e la il un a per in è una sono mi ho ma lo ha le si con non del della dei gli da più come anche questo "
          "quello perché nel nella delle degli molto tutti era stato essere hanno",
}
_SETS = {lang: set(words.split()) for lang, words in _WORDS.items()}
_COUNT = {}
for _w in _SETS.values():
    for _t in _w:
        _COUNT[_t] = _COUNT.get(_t, 0) + 1
_AMBIGUOUS = {t for t, n in _COUNT.items() if n > 1}  # palavras de mais de um idioma (a, o, de, como, sobre...) valem menos


def detect(text: str, min_words: int = 4) -> str | None:
    """Código do idioma ('pt', 'en', 'es', 'fr', 'de', 'it', 'ja', 'ko', 'zh', 'ru') ou None se não houver confiança."""
    if not text:
        return None
    if re.search(r"[぀-ヿ]", text):
        return "ja"
    if re.search(r"[가-힯]", text):
        return "ko"
    if re.search(r"[一-鿿]", text):
        return "zh"
    if re.search(r"[Ѐ-ӿ]", text):
        return "ru"
    tokens = re.findall(r"[^\W\d_]+", text.lower())
    if len(tokens) < min_words:
        return None
    def score(words: set) -> float:
        return sum((0.25 if t in _AMBIGUOUS else 1.0) for t in tokens if t in words)

    hits = sorted(((score(words), lang) for lang, words in _SETS.items()), reverse=True)
    (top, lang), (second, _) = hits[0], hits[1]
    if top < 1.5 or top < 1.4 * second:
        return None
    return lang


def same_language(text: str, target: str) -> bool | None:
    """True/False se o texto está/não está no idioma `target` (ex.: 'pt-BR'); None se não der para saber."""
    found = detect(text)
    if found is None:
        return None
    return found == (target or "").lower().split("-")[0]
