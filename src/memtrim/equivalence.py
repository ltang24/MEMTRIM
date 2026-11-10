import json

from . import prompts


def _resolve(client, messages):
    response = client.chat(messages, response_format=prompts.KEY_EQUIV_SCHEMA, max_tokens=64, temperature=0.0, seed=0)
    try:
        data = json.loads(response["content"])
    except (json.JSONDecodeError, TypeError):
        return None
    if not data.get("equivalent"):
        return None
    idx = data.get("candidate_index", -1)
    return idx if isinstance(idx, int) and idx >= 0 else None


def make_key_equiv_fn(client):
    def fn(entity, attribute, value_type, candidates):
        if not candidates:
            return None
        messages = prompts.build_key_equiv_messages(entity, attribute, value_type, candidates)
        return _resolve(client, messages)

    return fn


def make_value_equiv_fn(client):
    def fn(key_label, new_value, candidates):
        if not candidates:
            return None
        messages = prompts.build_value_equiv_messages(key_label, new_value, candidates)
        return _resolve(client, messages)

    return fn


def make_task_equiv_fn(client):
    def fn(operation, target, candidates):
        if not candidates:
            return None
        messages = prompts.build_task_equiv_messages(operation, target, candidates)
        return _resolve(client, messages)

    return fn
