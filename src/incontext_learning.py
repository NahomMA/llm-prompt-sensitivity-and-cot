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
        # Requires world knowledge: "paperweight" = useless device
        "review": "After the update it became the most expensive paperweight in my house.",
        "label": "Negative",
    },
    {
        # Every word is neutral/positive. Sentiment is entirely pragmatic.
        # "Replacement" implies the first one failed.
        "review": "The replacement works perfectly.",
        "label": "Negative",
    },
    {
        # Sounds like a complaint, but the reviewer is praising addictiveness
        "review": "Thanks to this thing I haven't slept properly in a week. I just can't put it down.",
        "label": "Positive",
    },
    {
        # Requires understanding that comparing a product to its
        # COMPETITOR favorably is still negative about THIS product
        "review": "It finally convinced me to switch to the competitor.",
        "label": "Negative",
    },
    {
        # Zero sarcasm markers. Deadpan. The "humor" is that the bar
        # is on the floor and the product barely clears it.
        "review": "It didn't catch fire. Exceeds expectations for this brand.",
        "label": "Negative",
    },
    {
        # Temporal reversal: present tense positive cancels past negative,
        # but the overall journey is positive because the USER changed.
        "review": "I left a one-star review six months ago. I was wrong.",
        "label": "Positive",
    },
    {
        # Every surface signal is negative. But the meaning:
        # the product is so good it made their old setup look bad.
        "review": "This ruined every other speaker I own. Nothing else sounds acceptable anymore.",
        "label": "Positive",
    },
    {
        # Conditional praise that actually reveals a fatal flaw.
        # "If you don't mind X" = X is a real problem.
        "review": "Fantastic product if you don't mind the smell of burning plastic every time you turn it on.",
        "label": "Negative",
    },
    {
        # Sounds generous (gift-giving) but implies product is
        # something you'd give away, not keep.
        "review": "I've been giving these away to everyone I know. Perfect gift for people I don't particularly like.",
        "label": "Negative",
    },
    {
        # Pure pragmatic implicature. "Still" doing something
        # basic after a long time = that's the whole point, it's reliable.
        # No sentiment words at all.
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




# ZERO_SHOT_TEMPLATE = """
# You are a sentiment analysis model.
# Classify the sentiment of the following short product review as exactly one word: Positive or Negative.

# Review: "{review}"

# Answer with exactly one word: Positive or Negative.
# """.strip()


# ONE_SHOT_TEMPLATE = """
# You are a sentiment analysis model.
# Use the example to infer the correct label.
# Respond with exactly one word: Positive or Negative.

# Example:
# Review: "This phone works great and the battery life is amazing."
# Label: Positive

# Now classify this review.
# Review: "{review}"
# Label:
# """.strip()


# FEW_SHOT_TEMPLATE = """
# You are a sentiment analysis model.
# Use the examples to infer the correct label.
# Respond with exactly one word: Positive or Negative.

# Examples:
# Review: "The headphones sound fantastic and are very comfortable."
# Label: Positive

# Review: "This camera is terrible, the pictures are always blurry."
# Label: Negative

# Review: "I love this laptop, it's fast and lightweight."
# Label: Positive

# Review: "The blender broke after one week of light use."
# Label: Negative

# Now classify this review.
# Review: "{review}"
# Label:
# """.strip()


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
    
  

    







