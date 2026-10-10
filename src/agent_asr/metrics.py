import difflib
import unicodedata

from rapidfuzz.distance import Levenshtein


def normalize(text, ignore_digits=False):
    text = unicodedata.normalize("NFKC", text).lower()
    digits = set("零〇一二两三四五六七八九十百千万亿点")
    return "".join(
        c
        for c in text
        if not c.isspace()
        and not unicodedata.category(c).startswith("P")
        and not (ignore_digits and (c.isdigit() or c in digits))
    )


def character_errors(reference, hypothesis, ignore_digits=False):
    reference = normalize(reference, ignore_digits)
    hypothesis = normalize(hypothesis, ignore_digits)
    if not reference:
        raise ValueError("Reference has no scoreable characters")
    distance = Levenshtein.distance(reference, hypothesis)
    return {
        "cer": distance / len(reference),
        "edit_distance": distance,
        "reference_characters": len(reference),
        "ignore_digits": ignore_digits,
        "normalization": "NFKC/lowercase/remove whitespace and punctuation; no simplified conversion",
    }


def text_diff(left, right):
    matcher = difflib.SequenceMatcher(a=left, b=right, autojunk=False)
    return [
        {"operation": op, "left": left[a:b], "right": right[c:d]}
        for op, a, b, c, d in matcher.get_opcodes()
        if op != "equal"
    ]
