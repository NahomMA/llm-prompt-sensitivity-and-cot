from typing import List, Dict


PROBLEMS: List[Dict[str, str]] = [
    {
        "question": "A store sells notebooks for $4 each. If you buy 3 or more, you get a 25% discount on the entire purchase. \
             How much do 5 notebooks cost?",
        "answer": "15",
    },
    {
        "question": "A train travels at 60 mph for 2.5 hours, then at 40 mph for 1.5 hours. What is the total distance traveled in miles?",
        "answer": "210",
    },
    {
        "question": "Tom is twice as old as Jerry. In 5 years, Tom will be 35. How old is Jerry now?",
        "answer": "15",
    },
    {
        "question": "A recipe calls for 2/3 cup of sugar for 4 servings. How many cups of sugar do you need for 18 servings?",
        "answer": "3",
    },
    {
        "question": "A pool fills at 12 gallons per minute and drains at 4 gallons per minute simultaneously. How many minutes does it take to accumulate 120 gallons?",
        "answer": "15",
    },
    {
        "question": "Three friends split a $84 dinner bill. They add a 20% tip before splitting. How much does each person pay in dollars?",
        "answer": "33.6",
    },
    {
        "question": "A shirt originally costs $60. It is marked down 30%, then an additional 10% off the sale price. What is the final price?",
        "answer": "37.8",
    },
    {
        "question": "You have 100 feet of fencing to make a rectangular garden. If the length is 10 feet more than the width, what is the area in square feet?",
        "answer": "600",
    },
    {
        "question": "A car uses 3 gallons of gas every 90 miles. Gas costs $4 per gallon. How much does gas cost for a 450-mile trip?",
        "answer": "60",
    },
    {
        "question": "A baker makes 48 cupcakes. She sells 1/3 on Monday, then half of what remains on Tuesday. How many are left?",
        "answer": "16",
    },
    {
        "question": "Two runners start from the same point running in opposite directions. One runs 7 mph and the other 5 mph. After how many hours are they 36 miles apart?",
        "answer": "3",
    },
    {
        "question": "A company's profit was $200,000 last year. It grew 15% this year but they paid 40% in taxes on this year's profit. What is the after-tax profit this year?",
        "answer": "138000",
    },
]


DIRECT_TEMPLATE = """
You are solving a short arithmetic word problem.
Read the problem carefully and provide only the final numeric answer.
Do not show your reasoning.

Problem: {question}

Answer (just the number):
""".strip()


COT_TEMPLATE = """
You are solving a short arithmetic word problem.
First, think through the problem step by step.
Then, on the last line, provide only the final numeric answer.

Problem: {question}

Let's reason step by step, then give the final answer.
""".strip()


class ChainOfThoughtTemplates:
    """
    Helper class to build direct-answer and chain-of-thought prompts.
    """

    def __init__(self) -> None:
        self.direct_template = DIRECT_TEMPLATE
        self.cot_template = COT_TEMPLATE

    def get_prompt(self, style: str, question: str) -> str:
        if style == "direct":
            return self.direct_template.format(question=question)
        if style == "cot":
            return self.cot_template.format(question=question)
        raise ValueError(f"Unknown prompting style: {style}")