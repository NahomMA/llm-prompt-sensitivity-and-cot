from datasets import load_dataset
from typing import Dict, Tuple, List
from config import config
from src.incontext_learning import IncontextLearning_Templates, REVIEWS
from src.classifier import OpenAIModel, GeminiModel
from src.cot import PROBLEMS, ChainOfThoughtTemplates
from src.evaluator import Evaluator

POS_LABEL = "positive"
NEG_LABEL = "negative"


def extract_label(text: str) -> str:
    """Extract 'positive' or 'negative' from a model response."""
    if not text:
        return "unknown"

    text_lower = text.strip().lower()

    # If the model already returned a single word, normalize it.
    first_word = text_lower.split()[0]
    if first_word in {POS_LABEL, NEG_LABEL}:
        return first_word

    pos_index = text_lower.find(POS_LABEL)
    neg_index = text_lower.find(NEG_LABEL)

    if pos_index == -1 and neg_index == -1:
        return "unknown"
    elif pos_index != -1:
        return POS_LABEL
    elif neg_index != -1:
        return NEG_LABEL

    # Both appear: choose whichever appears first.
    return POS_LABEL if pos_index < neg_index else NEG_LABEL


def incontext_learning(num_examples: int = 10) -> None:
    """
    Problem 1:
    - Use the custom list of 10 short product reviews in REVIEWS.
    - Apply zero-shot, one-shot, and few-shot prompts to the same set of reviews.
    - Evaluate OpenAI (GPT-4.1) and Gemini models.
    """
    iclt = IncontextLearning_Templates()
    samples = REVIEWS[:num_examples]

    openai_model = OpenAIModel()
    gemini_model = GeminiModel()

    models = {
        "openai_gpt4.1": openai_model,
        "gemini_2.0_flash": gemini_model,
    }
    strategies = ["zero", "one", "few"]

    # (model, strategy) -> example_id -> prediction label
    results: Dict[Tuple[str, str], Dict[str, str]] = {}

    for model_name in models:
        for strategy in strategies:
            results[(model_name, strategy)] = {}

    for idx, sample in enumerate(samples):
        text = sample["review"]
        gold = sample["label"].lower()

        print(f"\n=== Example {idx} ===")
        print(f"Review: {text}")
        print(f"Gold label: {gold}")

        for model_name, model in models.items():
            for strategy in strategies:
                prompt = iclt.get_prompt(strategy, text)
                raw_output = model.generate(prompt)
                pred = extract_label(raw_output)
                results[(model_name, strategy)][f"example_{idx}"] = pred
                print(
                    f"{model_name} | {strategy}-shot -> pred: {pred} "
                    f"(raw: {raw_output!r})"
                )

    print("\n=== Accuracy summary (Problem 1) ===")
    for model_name in models:
        for strategy in strategies:
            preds = results[(model_name, strategy)]
            correct = 0
            for idx, sample in enumerate(samples):
                gold = sample["label"].lower()
                pred = preds.get(f"example_{idx}", "unknown")
                if pred == gold:
                    correct += 1
            acc = correct / len(samples)
            print(
                f"{model_name} | {strategy}-shot: "
                f"{correct}/{len(samples)} correct, accuracy = {acc:.2f}"
            )

def normalize_numeric_answer(text: str) -> str:
    if not text:
        return ""
    text = text.strip()
    # Take the last line and try to grab the first integer in it.
    last_line = text.splitlines()[-1]
    tokens = last_line.replace(",", " ").split()
    for tok in tokens:
        if tok.lstrip("+-").isdigit():
            return tok.lstrip("+")
    # Fallback: if the whole text is a number
    if text.lstrip("+-").isdigit():
        return text.lstrip("+")
    return text


def chain_of_thought_prompting() -> None:   
    templates = ChainOfThoughtTemplates()
    evaluator = Evaluator()

    openai_model = OpenAIModel()
    gemini_model = GeminiModel()

    models = {
        "openai_gpt4.1": openai_model,
        "gemini_flash": gemini_model,
    }

    styles = ["direct", "cot"]

    ground_truths: List[str] = [p["answer"] for p in PROBLEMS]
    predictions: Dict[str, Dict[str, List[str]]] = {
        model_name: {style: [] for style in styles} for model_name in models
    }

    print("\n=== Problem 2: Chain-of-thought vs direct-answer ===")
    for idx, problem in enumerate(PROBLEMS):
        question = problem["question"]
        actual_answer = problem["answer"]

        print(f"\n--- Problem {idx} ---")
        print(f"Question: {question}")
        print(f"Actual answer: {actual_answer}")

        for model_name, model in models.items():
            for style in styles:
                prompt = templates.get_prompt(style, question)
                output = model.generate(prompt)
                pred = normalize_numeric_answer(output)
                predictions[model_name][style].append(pred)
                print(
                    f"{model_name} | {style} -> pred: {pred!r} "
                    f"(raw: {output[:200]!r}...)"
                )

    print("\n=== Accuracy summary (Problem 2) ===")
    for model_name in models:
        for style in styles:
            preds = predictions[model_name][style]
            cm, acc, prec, rec, f1 = evaluator.evaluate(preds, ground_truths)
            print(f"\nModel: {model_name} | Style: {style}")
            print(f"Accuracy: {acc:.2f}")
            print(f"Precision (macro): {prec:.2f}")
            print(f"Recall (macro): {rec:.2f}")
            print(f"F1 (macro): {f1:.2f}")
            print("Confusion matrix:")
            print(cm)


if __name__ == "__main__":
    # Run Problem 1 (in-context learning) and Problem 2 (CoT) as needed.
    incontext_learning()
    # chain_of_thought_prompting()

