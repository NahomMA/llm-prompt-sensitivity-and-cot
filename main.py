import os
import csv
from pathlib import Path
from datasets import load_dataset
from typing import Dict, Tuple, List
from config import config
from src.incontext_learning import IncontextLearning_Templates, REVIEWS, SENSITIVITY_VARIANTS
from src.models import OpenAIModel, GeminiModel, AnthropicModel
from src.cot import PROBLEMS, ChainOfThoughtTemplates
from src.evaluator import Evaluator

POS_LABEL = "positive"
NEG_LABEL = "negative"


def extract_label(text: str) -> str:
    if not text:
        return "unknown"

    text_lower = text.strip().lower()

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

    return POS_LABEL if pos_index < neg_index else NEG_LABEL


def incontext_learning(num_examples: int = 10) -> None:  
    iclt = IncontextLearning_Templates()
    samples = REVIEWS[:num_examples]

    # openai_model = OpenAIModel()
    gemini_model = GeminiModel()
    anthropic_model = AnthropicModel()

    models = {
        # "openai_gpt4.1": openai_model,
        "gemini_2.5_flash": gemini_model,
        "anthropic_haiku_4.5": anthropic_model,
    }
    strategies = ["zero", "one", "few"]

    results: Dict[Tuple[str, str], Dict[str, str]] = {}

    for model_name in models:
        for strategy in strategies:
            results[(model_name, strategy)] = {}

    # store detailed predictions for saving
    detailed_rows: List[Dict[str, str]] = []

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
                key = f"example_{idx}"
                results[(model_name, strategy)][key] = pred
                detailed_rows.append(
                    {
                        "example_id": key,
                        "review": text,
                        "gold_label": gold,
                        "model": model_name,
                        "strategy": strategy,
                        "prediction": pred,
                        "raw_output": raw_output,
                    }
                )
                print(
                    f"{model_name} | {strategy}-shot -> pred: {pred} "
                    f"(raw: {raw_output!r})"
                )

    print("\n=== Accuracy summary (Problem 1) ===")
    summary_rows: List[Dict[str, str]] = []
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
            summary_rows.append(
                {
                    "model": model_name,
                    "strategy": strategy,
                    "correct": str(correct),
                    "total": str(len(samples)),
                    "accuracy": f"{acc:.4f}",
                }
            )

    # save results to CSV files under results/
    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)

    detailed_path = results_dir / "problem1_predictions.csv"
    summary_path = results_dir / "problem1_summary.csv"

    if detailed_rows:
        with detailed_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "example_id",
                    "review",
                    "gold_label",
                    "model",
                    "strategy",
                    "prediction",
                    "raw_output",
                ],
            )
            writer.writeheader()
            writer.writerows(detailed_rows)

    if summary_rows:
        with summary_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["model", "strategy", "correct", "total", "accuracy"],
            )
            writer.writeheader()
            writer.writerows(summary_rows)


def run_prompt_sensitivity(num_examples: int = 10) -> None:
    """Problem 3: Run baseline (few-shot) and perturbed prompts on same 10 reviews, two models."""
    iclt = IncontextLearning_Templates()
    samples = REVIEWS[:num_examples]
    variants = list(SENSITIVITY_VARIANTS.keys())

    gemini_model = GeminiModel()
    anthropic_model = AnthropicModel()
    models = {
        "gemini_2.5_flash": gemini_model,
        "anthropic_haiku_4.5": anthropic_model,
    }

    results: Dict[Tuple[str, str], Dict[str, str]] = {}
    for model_name in models:
        for variant in variants:
            results[(model_name, variant)] = {}

    detailed_rows: List[Dict[str, str]] = []

    print("\n=== Problem 3: Prompt sensitivity (few-shot baseline + perturbations) ===")
    for idx, sample in enumerate(samples):
        text = sample["review"]
        gold = sample["label"].lower()
        for model_name, model in models.items():
            for variant in variants:
                prompt = iclt.get_sensitivity_prompt(variant, text)
                raw_output = model.generate(prompt)
                pred = extract_label(raw_output)
                key = f"example_{idx}"
                results[(model_name, variant)][key] = pred
                detailed_rows.append({
                    "example_id": key,
                    "review": text,
                    "gold_label": gold,
                    "model": model_name,
                    "variant": variant,
                    "prediction": pred,
                    "raw_output": raw_output,
                })
                print(f"  {model_name} | {variant} | ex{idx} -> {pred}")

    print("\n=== Accuracy summary (Problem 3) ===")
    summary_rows: List[Dict[str, str]] = []
    for model_name in models:
        for variant in variants:
            preds = results[(model_name, variant)]
            correct = sum(1 for i, s in enumerate(samples) if preds.get(f"example_{i}", "") == s["label"].lower())
            acc = correct / len(samples)
            summary_rows.append({
                "model": model_name,
                "variant": variant,
                "correct": str(correct),
                "total": str(len(samples)),
                "accuracy": f"{acc:.4f}",
            })
            print(f"{model_name} | {variant}: {correct}/{len(samples)} correct, accuracy = {acc:.2f}")

    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)
    if detailed_rows:
        with (results_dir / "problem3_predictions.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["example_id", "review", "gold_label", "model", "variant", "prediction", "raw_output"])
            w.writeheader()
            w.writerows(detailed_rows)
    if summary_rows:
        with (results_dir / "problem3_summary.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["model", "variant", "correct", "total", "accuracy"])
            w.writeheader()
            w.writerows(summary_rows)


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

    # openai_model = OpenAIModel()
    gemini_model = GeminiModel()
    anthropic_model = AnthropicModel()
    models = {
        # "openai_gpt4.1": openai_model,
        "gemini_2.5_flash": gemini_model,
        "anthropic_haiku_4.5": anthropic_model,
    }

    styles = ["direct", "cot"]

    ground_truths: List[str] = [p["answer"] for p in PROBLEMS]
    predictions: Dict[str, Dict[str, List[str]]] = {
        model_name: {style: [] for style in styles} for model_name in models
    }

    print("\n=== Problem 2: Chain-of-thought vs direct-answer ===")
    # for saving detailed outputs
    detailed_rows: List[Dict[str, str]] = []
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
                detailed_rows.append(
                    {
                        "problem_id": str(idx),
                        "question": question,
                        "gold_answer": actual_answer,
                        "model": model_name,
                        "style": style,
                        "prediction": pred,
                        "raw_output": output,
                    }
                )
                print(
                    f"{model_name} | {style} -> pred: {pred!r} "
                    f"(raw: {output[:200]!r}...)"
                )

    print("\n=== Accuracy summary (Problem 2) ===")
    summary_rows: List[Dict[str, str]] = []
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
            summary_rows.append(
                {
                    "model": model_name,
                    "style": style,
                    "accuracy": f"{acc:.4f}",
                    "precision_macro": f"{prec:.4f}",
                    "recall_macro": f"{rec:.4f}",
                    "f1_macro": f"{f1:.4f}",
                }
            )

    #save Problem 2 results
    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)

    detailed_path = results_dir / "problem2_predictions.csv"
    summary_path = results_dir / "problem2_summary.csv"

    if detailed_rows:
        with detailed_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "problem_id",
                    "question",
                    "gold_answer",
                    "model",
                    "style",
                    "prediction",
                    "raw_output",
                ],
            )
            writer.writeheader()
            writer.writerows(detailed_rows)

    if summary_rows:
        with summary_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "model",
                    "style",
                    "accuracy",
                    "precision_macro",
                    "recall_macro",
                    "f1_macro",
                ],
            )
            writer.writeheader()
            writer.writerows(summary_rows)


if __name__ == "__main__":
    # incontext_learning()
    # chain_of_thought_prompting()
    run_prompt_sensitivity()

