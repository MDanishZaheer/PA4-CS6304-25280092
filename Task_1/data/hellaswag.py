# Name : Muhammad Danish Zaheer Awan
# REG id : 25280092

"""Lightweight HellaSwag formatting shared by compute and audited replay."""

import re


def format_hellaswag(example):
    """Match HellaSwag v1.0 text/choice behavior documented by harness v0.4.9."""
    def clean(text):
        """Remove WikiHow formatting artifacts without changing character units."""
        value = text.strip().replace(" [title]", ". ")
        return re.sub(r"\[.*?\]", "", value).replace("  ", " ")

    context = example["ctx_a"] + " " + example["ctx_b"].capitalize()
    prompt = clean(example["activity_label"] + ": " + context)
    endings = [clean(ending) for ending in example["endings"]]
    if len(endings) != 4 or any(not ending for ending in endings):
        raise ValueError("HellaSwag requires four nonempty preprocessed endings.")
    return prompt, endings, int(example["label"])
