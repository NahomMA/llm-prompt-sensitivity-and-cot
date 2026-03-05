from openai import OpenAI
from dotenv import load_dotenv
import os
from typing import Dict, Tuple, List
from config import config
from datasets import load_dataset
import google.generativeai as genai

load_dotenv()


REVIEWS: List[Dict[str, str]] = [
    {
        "review": "After the update it became the most expensive paperweight in my house.",
        "label": "Negative",
    },
    {
        "review": "The replacement works perfectly.",
        "label": "Negative",
    },
    {
        "review": "Thanks to this thing I haven't slept properly in a week. I just can't put it down.",
        "label": "Positive",
    },
    {
        "review": "It finally convinced me to switch to the competitor.",
        "label": "Negative",
    },
    {
        "review": "It didn't catch fire. Exceeds expectations for this brand.",
        "label": "Negative",
    },
    {
        "review": "I left a one-star review six months ago. I was wrong.",
        "label": "Positive",
    },
    {
        "review": "This ruined every other speaker I own. Nothing else sounds acceptable anymore.",
        "label": "Positive",
    },
    {
        "review": "Fantastic product if you don't mind the smell of burning plastic every time you turn it on.",
        "label": "Negative",
    },
    {
        "review": "I've been giving these away to everyone I know. Perfect gift for people I don't particularly like.",
        "label": "Negative",
    },
    {
        "review": "Bought this four years ago. Still here. Still works. Nothing else to say.",
        "label": "Positive",
    },
]


# ── Prompt Templates ──────────────────────────────────────────────────

ZERO_SHOT_TEMPLATE = """
You are a sentiment classifier for product reviews.
Some reviews use sarcasm, irony, or mixed language. Focus on the reviewer's true overall intent, not surface-level word choice.
Classify the review as exactly one word: Positive or Negative.

Review: "{review}"

Label:
""".strip()


ONE_SHOT_TEMPLATE = """
You are a sentiment classifier for product reviews.
Some reviews use sarcasm, irony, or mixed language. Focus on the reviewer's true overall intent.
Classify the review as exactly one word: Positive or Negative.

Example:
Review: "Congratulations to the manufacturer for making something that breaks in record time. Truly an achievement."
Label: Negative

Now classify:
Review: "{review}"
Label:
""".strip()


FEW_SHOT_TEMPLATE = """
You are a sentiment classifier for product reviews.
Some reviews use sarcasm, irony, or mixed language. Focus on the reviewer's true overall intent.
Classify the review as exactly one word: Positive or Negative.

Examples:

Review: "Oh great, another charger that works for exactly one month. Money well spent."
Label: Negative

Review: "I was skeptical after the bad reviews but honestly this thing exceeded every expectation I had."
Label: Positive

Review: "It's not pretty and the instructions were useless, but once it's set up it runs like a dream."
Label: Positive

Review: "Sure, it looks nice on the shelf. Too bad it doesn't actually do what it's supposed to."
Label: Negative

Review: "I bought three more for my family — that should tell you everything."
Label: Positive

Now classify:
Review: "{review}"
Label:
""".strip()


# Problem 3: Sensitivity variants (baseline = few-shot, then perturbations) 

# Baseline: same as FEW_SHOT_TEMPLATE (original order: Neg, Pos, Pos, Neg, Pos)
SENSITIVITY_BASELINE = FEW_SHOT_TEMPLATE

# Reorder: put Positive examples first, then Negative (Pos, Pos, Pos, Neg, Neg)
SENSITIVITY_REORDER = """
You are a sentiment classifier for product reviews.
Some reviews use sarcasm, irony, or mixed language. Focus on the reviewer's true overall intent.
Classify the review as exactly one word: Positive or Negative.

Examples:

Review: "I was skeptical after the bad reviews but honestly this thing exceeded every expectation I had."
Label: Positive

Review: "It's not pretty and the instructions were useless, but once it's set up it runs like a dream."
Label: Positive

Review: "I bought three more for my family — that should tell you everything."
Label: Positive

Review: "Oh great, another charger that works for exactly one month. Money well spent."
Label: Negative

Review: "Sure, it looks nice on the shelf. Too bad it doesn't actually do what it's supposed to."
Label: Negative

Now classify:
Review: "{review}"
Label:
""".strip()

# Rephrase: shorter instruction line (no "exactly one word")
SENSITIVITY_REPHRASE = """
You are a sentiment classifier for product reviews.
Some reviews use sarcasm or irony; focus on the reviewer's true intent.
Classify the sentiment as Positive or Negative.

Examples:

Review: "Oh great, another charger that works for exactly one month. Money well spent."
Label: Negative

Review: "I was skeptical after the bad reviews but honestly this thing exceeded every expectation I had."
Label: Positive

Review: "It's not pretty and the instructions were useless, but once it's set up it runs like a dream."
Label: Positive

Review: "Sure, it looks nice on the shelf. Too bad it doesn't actually do what it's supposed to."
Label: Negative

Review: "I bought three more for my family — that should tell you everything."
Label: Positive

Now classify:
Review: "{review}"
Label:
""".strip()

# Add constraint: explicit "do not explain"
SENSITIVITY_ADD_CONSTRAINT = """
You are a sentiment classifier for product reviews.
Some reviews use sarcasm, irony, or mixed language. Focus on the reviewer's true overall intent.
Classify the review as exactly one word: Positive or Negative.
Do not explain; output only the label.

Examples:

Review: "Oh great, another charger that works for exactly one month. Money well spent."
Label: Negative

Review: "I was skeptical after the bad reviews but honestly this thing exceeded every expectation I had."
Label: Positive

Review: "It's not pretty and the instructions were useless, but once it's set up it runs like a dream."
Label: Positive

Review: "Sure, it looks nice on the shelf. Too bad it doesn't actually do what it's supposed to."
Label: Negative

Review: "I bought three more for my family — that should tell you everything."
Label: Positive

Now classify:
Review: "{review}"
Label:
""".strip()

# Remove constraint: drop "exactly one word"
SENSITIVITY_REMOVE_CONSTRAINT = """
You are a sentiment classifier for product reviews.
Some reviews use sarcasm, irony, or mixed language. Focus on the reviewer's true overall intent.
Classify the review as Positive or Negative.

Examples:

Review: "Oh great, another charger that works for exactly one month. Money well spent."
Label: Negative

Review: "I was skeptical after the bad reviews but honestly this thing exceeded every expectation I had."
Label: Positive

Review: "It's not pretty and the instructions were useless, but once it's set up it runs like a dream."
Label: Positive

Review: "Sure, it looks nice on the shelf. Too bad it doesn't actually do what it's supposed to."
Label: Negative

Review: "I bought three more for my family — that should tell you everything."
Label: Positive

Now classify:
Review: "{review}"
Label:
""".strip()

SENSITIVITY_VARIANTS: Dict[str, str] = {
    "baseline": SENSITIVITY_BASELINE,
    "reorder": SENSITIVITY_REORDER,
    "rephrase": SENSITIVITY_REPHRASE,
    "add_constraint": SENSITIVITY_ADD_CONSTRAINT,
    "remove_constraint": SENSITIVITY_REMOVE_CONSTRAINT,
}


class IncontextLearning_Templates:
    def __init__(self) -> None:
        self.zero_shot_template = ZERO_SHOT_TEMPLATE
        self.one_shot_template = ONE_SHOT_TEMPLATE
        self.few_shot_template = FEW_SHOT_TEMPLATE

    def get_prompt(self, strategy: str, review: str) -> str:
        if strategy == "few":
            return self.few_shot_template.format(review=review)
        if strategy == "one":
            return self.one_shot_template.format(review=review)
        if strategy == "zero":
            return self.zero_shot_template.format(review=review)
        raise ValueError(f"Unknown strategy: {strategy}")

    def get_sensitivity_prompt(self, variant: str, review: str) -> str:
        if variant not in SENSITIVITY_VARIANTS:
            raise ValueError(f"Unknown sensitivity variant: {variant}")
        return SENSITIVITY_VARIANTS[variant].format(review=review)








