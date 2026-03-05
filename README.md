# LLM Prompting Strategies and Their Effect on Model Behavior

CS 895: LLM Architecture & Applications — Homework 2

## Overview

This project investigates how prompting strategies affect large language model behavior across three experiments using Gemini 2.5 Flash and Claude Haiku.

**Problem 1 — In-Context Learning:** Zero-shot, one-shot, and few-shot sentiment classification on 10 hand-crafted sarcastic product reviews. Tests whether adding labeled examples improves accuracy on pragmatically difficult cases like irony and deadpan negativity.

**Problem 2 — Chain-of-Thought Prompting:** Direct-answer vs. CoT prompting on 12 arithmetic word problems. Compares whether explicit reasoning steps improve or hurt accuracy across models.

**Problem 3 — Prompt Sensitivity:** Eight controlled perturbations of the few-shot sentiment prompt (example reordering, instruction rephrasing, constraint addition/removal, fewer examples, contradictory labels, role change, neutral label expansion). Measures robustness of both models to small prompt changes.

## Project Structure

```
├── main.py                      # Entry point, runs all three experiments
├── src/
│   ├── incontext_learning.py    # Prompt templates and sensitivity variants
│   ├── cot.py      # Direct-answer and CoT templates
│   ├── models.py                # Gemini and Claude API wrappers
│   └── evaluation.py            # Accuracy, precision, recall, F1 computation
├── results/                     # Generated CSV outputs
│   ├── problem1_predictions.csv
│   ├── problem1_summary.csv
│   ├── problem2_predictions.csv
│   ├── problem2_summary.csv
│   ├── problem3_predictions.csv
│   └── problem3_summary.csv
├── .env                         
├── pyproject.toml
└── README.md
```

## Setup

Requires Python 3.10+ and [uv](https://github.com/astral-sh/uv).

```bash
git clone <repo-url>
cd <repo>
uv sync
```

Create a `.env` file in the project root:

```
GEMINI_API_KEY=your_gemini_api_key
ANTHROPIC_API_KEY=your_anthropic_api_key
```

## Usage

Run all three experiments:

```bash
uv run main.py
```

Results are written as CSV files under `results/`.

## Key Findings

| Experiment | Finding |
|---|---|
| In-context learning | Both models reached 0.80 zero-shot and 0.90 one-shot. Few-shot kept Gemini at 0.90 but dropped Claude to 0.80. |
| Chain-of-thought | CoT hurt Gemini (0.83 → 0.75) by introducing arithmetic errors. Claude stayed at 0.75 but CoT improved its precision and F1. |
| Prompt sensitivity | Gemini held at 0.90 for 4/8 variants. Claude dropped to 0.70 on constraint removal but rose to 0.90 with fewer examples. |

## Models Used

- **Gemini 2.5 Flash** (Google)
- **Claude Haiku** (Anthropic)

## References

- Brown et al. (2020). [Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165). NeurIPS.
- Wei et al. (2022). [Chain-of-Thought Prompting Elicits Reasoning in Large Language Models](https://arxiv.org/abs/2201.11903). NeurIPS.
- Yao et al. (2023). [ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629). ICLR.