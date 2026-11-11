import re
import unicodedata

_LEADING_ARTICLE_RE = re.compile(r"^(the|a|an)\s+")
_PUNCT_RE = re.compile(r"[^\w\s]")
_SPACE_RE = re.compile(r"\s+")


def normalize(text):
    if text is None:
        return ""
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = text.lower().strip()
    text = _PUNCT_RE.sub(" ", text)
    text = _SPACE_RE.sub(" ", text).strip()
    text = _LEADING_ARTICLE_RE.sub("", text)
    return text


def gold_forms(value, aliases=None):
    forms = {normalize(value)}
    for a in aliases or []:
        forms.add(normalize(a))
    return {f for f in forms if f}


def contains_gold(prediction_text, gold_forms_set):
    norm_pred = normalize(prediction_text)
    if not norm_pred:
        return False
    for form in gold_forms_set:
        if not form:
            continue
        pattern = r"(?<!\w)" + re.escape(form) + r"(?!\w)"
        if re.search(pattern, norm_pred):
            return True
    return False


def forms_equal(forms_a, forms_b):
    return len(forms_a & forms_b) > 0
