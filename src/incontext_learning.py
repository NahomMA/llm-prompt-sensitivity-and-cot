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


# Problem 3: Sensitivity variants

# Baseline: same as FEW_SHOT_TEMPLATE (original order: Neg, Pos, Pos, Neg, Pos)
SENSITIVITY_BASELINE = FEW_SHOT_TEMPLATE

# --- VARIANT 1: Reorder (Positive-first bias) ---
# All positive examples first, then negative. Tests recency bias (last examples
# may anchor the model's prior toward Negative).
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

# --- VARIANT 2: Rephrase system instruction ---
# Shorter, less prescriptive instruction. No mention of sarcasm or irony,
# removing the hint that helps with tricky reviews.
SENSITIVITY_REPHRASE = """
Classify product review sentiment. Output only: Positive or Negative.

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

# --- VARIANT 3: Add constraint (explicit reasoning prohibition + format lock) ---
# Forces single-token output and prohibits hedging. May cause the model to
# commit to wrong answers on ambiguous reviews instead of reasoning through them.
SENSITIVITY_ADD_CONSTRAINT = """
You are a sentiment classifier for product reviews.
Some reviews use sarcasm, irony, or mixed language. Focus on the reviewer's true overall intent.
Classify the review as exactly one word: Positive or Negative.
Do not explain, hedge, or add any other text. Your entire response must be one word.

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

# --- VARIANT 4: Remove constraint (open-ended, no format guidance) ---
# No role assignment, no format instruction, no sarcasm hint. Just examples.
# Tests whether the model can infer the task purely from demonstrations.
SENSITIVITY_REMOVE_CONSTRAINT = """
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

Review: "{review}"
Label:
""".strip()

# --- VARIANT 5: Reduce to 2 examples (minimal few-shot) ---
# Only 1 positive and 1 negative example. Dramatically reduces the model's
# ability to learn the boundary from demonstrations, especially for sarcasm.
SENSITIVITY_FEWER_EXAMPLES = """
You are a sentiment classifier for product reviews.
Some reviews use sarcasm, irony, or mixed language. Focus on the reviewer's true overall intent.
Classify the review as exactly one word: Positive or Negative.

Examples:

Review: "Oh great, another charger that works for exactly one month. Money well spent."
Label: Negative

Review: "I was skeptical after the bad reviews but honestly this thing exceeded every expectation I had."
Label: Positive

Now classify:
Review: "{review}"
Label:
""".strip()

# --- VARIANT 6: Contradictory label (deliberately mislabel one example) ---
# One example has a WRONG label. Tests whether the model follows the
# (incorrect) demonstration or overrides with its own judgment.
SENSITIVITY_CONTRADICTORY = """
You are a sentiment classifier for product reviews.
Some reviews use sarcasm, irony, or mixed language. Focus on the reviewer's true overall intent.
Classify the review as exactly one word: Positive or Negative.

Examples:

Review: "Oh great, another charger that works for exactly one month. Money well spent."
Label: Positive

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

# --- VARIANT 7: Opposite role framing ---
# Frames the model as a marketing analyst looking for purchase intent rather
# than a sentiment classifier. Same examples, different cognitive frame.
SENSITIVITY_ROLE_CHANGE = """
You are a marketing analyst evaluating whether product reviews indicate customer satisfaction.
Classify each review as exactly one word: Positive or Negative.

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

# --- VARIANT 8: Add neutral option (expand label space) ---
# Introduces a third class. The model may now classify ambiguous or
# mixed-sentiment reviews as Neutral, splitting what was previously forced
# into Positive or Negative. Will likely drop accuracy on the binary ground truth.
SENSITIVITY_ADD_NEUTRAL = """
You are a sentiment classifier for product reviews.
Some reviews use sarcasm, irony, or mixed language. Focus on the reviewer's true overall intent.
Classify the review as exactly one word: Positive, Negative, or Neutral.

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
    "fewer_examples": SENSITIVITY_FEWER_EXAMPLES,
    "contradictory": SENSITIVITY_CONTRADICTORY,
    "role_change": SENSITIVITY_ROLE_CHANGE,
    "add_neutral": SENSITIVITY_ADD_NEUTRAL,
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








