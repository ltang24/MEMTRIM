import re

from . import alias

_YES_NO_RE = re.compile(r"\b(yes|no)\b")


def first_yes_no(text):
    match = _YES_NO_RE.search(alias.normalize(text))
    return match.group(1) if match else None


def is_correct(prediction, gold, aliases=None, answer_type="entity"):
    if prediction is None:
        return False
    if answer_type == "yes_no":
        return first_yes_no(prediction) == alias.normalize(gold)
    return alias.contains_gold(prediction, alias.gold_forms(gold, aliases))
