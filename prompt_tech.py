from openai import OpenAI
from dotenv import load_dotenv
import os
from config import config
from datasets import load_dataset

load_dotenv()


# client = OpenAI(api_key=oppenai_api_key)

class IncontextLearning:
    def __init__(self, api_key: str):
        self.api_key = os.getenv(config["api_key_env"])
        self.model = config["model"]
        self.temperature = config['temperature']        
        self.max_tokens = config['max_tokens']
        self.client = OpenAI(api_key=self.api_key)

    def zero_shot_learning(self, prompt: str, sample_data: list[str]) -> str:
        return self.client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            messages=[{"role": "user", "content": prompt}, {"role": "assistant", "content": sample_data}],)

    def one_shot_learning(self, prompt: str, sample_data: list[str]) -> str:
        return self.client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            messages=[{"role": "user", "content": prompt}],)

    def few_shot_learning(self, prompt: str, sample_data: list[str]) -> str:
        return self.client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens, 
 
 
            messages=[{"role": "user", "content": prompt}],)



def run():
    ds = load_dataset("stanfordnlp/sst2", split="validation")
    for sample in ds[:10]:
        print(f"sample: {sample}")
    # incontext_learning = IncontextLearning()
    #   #prob 1 a
    # incontext_learning.zero_shot_learning("What is the capital of France?")


    # #prob 1 b
    # incontext_learning.one_shot_learning("What is the capital of France?", ["Paris"])


    # #prob 1 c
    # incontext_learning.few_shot_learning("What is the capital of France?", ["Paris"])

    #prob 1 d

    #prob 1 e


if __name__ == "__main__":
    run()