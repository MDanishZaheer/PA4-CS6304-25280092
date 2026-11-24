# Name : Muhammad Danish Zaheer Awan
# REG id : 25280092

"""Separate answer-appropriate standalone generation from short speed-test continuations."""

import re
from decimal import Decimal, InvalidOperation

from Task_3.common.artifacts import read_json


def numeric_answer(text, gold=False):
    """Use GSM8K's #### marker when present, otherwise the final numeric expression."""
    selected = text.rsplit("####", 1)[-1] if "####" in text else text
    boxed = re.findall(r"\\boxed\{([^{}]+)\}", selected)
    if boxed:
        selected = boxed[-1]
    matches = re.findall(r"[-+]?\d[\d,]*(?:\.\d+)?", selected)
    if not matches:
        return None
    try:
        return str(Decimal(matches[-1].replace(",", "")).normalize())
    except InvalidOperation:
        return None


def score_answer(prompt, text):
    """Keep exact correctness, extraction success and format compliance distinct."""
    if prompt["domain"] == "math":
        answer, gold = numeric_answer(text), numeric_answer(prompt["answer"], True)
        return {"extracted_answer": answer, "correct": answer is not None and answer == gold,
                "extractable": answer is not None, "format_compliant": answer is not None}
    stripped = text.strip()
    if prompt["family"] == 0:
        found = re.findall(r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\b", text, re.I)
        extracted = found[-1].lower() if found else None
        compliant = re.fullmatch(r"January|February|March|April|May|June|July|August|September|October|November|December", stripped, re.I) is not None
    elif prompt["family"] == 1:
        found = re.findall(r"\b\d+\b", text)
        extracted = found[-1] if found else None
        compliant = re.fullmatch(r"\d+", stripped) is not None
    else:
        found = re.findall(r"\b[A-Z]{2}\b", text)
        extracted = found[-1].lower() if found else None
        compliant = re.fullmatch(r"[A-Z]{2}", stripped) is not None
    return {"extracted_answer": extracted, "correct": extracted == prompt["answer"].lower() if extracted is not None else False,
            "extractable": extracted is not None, "format_compliant": compliant}


def evaluate_quality(store, get_model, tokenizer, prompts, names):
    """Evaluate the five standalone drafts and a separate target reference at proper caps."""
    from Task_3.sampling.exact import target_generate
    artifacts = []
    for name in names:
        model = get_model(name)
        for repetition in range(store.configuration["sampling"]["repetitions"]):
            for index, prompt in enumerate(prompts):
                store.check_time()
                relative = f"quality/{name}/rep{repetition}/{prompt['id']}.json"
                if not store.path(relative).is_file():
                    seed = 6304 + 10007 * index + 1000003 * repetition
                    response = target_generate(model, prompt["prompt_token_ids"],
                        store.configuration["sampling"]["quality_caps"][prompt["domain"]], seed)
                    text = tokenizer.decode(response["token_ids"], skip_special_tokens=True)
                    store.save_json(relative, {"prompt_id": prompt["id"], "prompt_sha256": prompt["prompt_sha256"],
                        "draft": name, "workload": prompt["workload"], "repetition": repetition, "response": response,
                        "text": text, **score_answer(prompt, text), "evaluation": "standalone model quality; not speculative-output quality"})
                row = read_json(store.path(relative))
                if row["prompt_sha256"] != prompt["prompt_sha256"]:
                    raise ValueError("Standalone quality prompt differs from its frozen encoding.")
                artifacts.append(relative)
    store.complete("quality", artifacts, {"models": len(names), "prompts": len(prompts), "repetitions": 2})
    return {"records": len(artifacts), "answer_caps": store.configuration["sampling"]["quality_caps"]}
