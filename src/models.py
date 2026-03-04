from openai import OpenAI
from dotenv import load_dotenv
import os
from typing import Dict, Tuple
from config import config
import google.generativeai as genai
from anthropic import Anthropic


load_dotenv()


class OpenAIModel:
    def __init__(self) -> None:
        api_key = os.getenv(config["api_key_env"])
        if not api_key:
            raise RuntimeError(f"{config['api_key_env']} is not set in the environment.")
        self.client = OpenAI(api_key=api_key)
        self.model = config["openai_model"]
        self.temperature = config["temperature"]
        self.max_tokens = config["max_tokens"]

    def generate(self, prompt: str) -> str:
        completion = self.client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return completion.choices[0].message.content.strip()


class AnthropicModel:
    def __init__(self, model_name: str = config["anthropic_model"]) -> None:
        api_key = os.getenv(config["anthropic_api_key"])
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY is not set in the environment.")
        self.client = Anthropic(api_key=api_key)
        self.model = model_name
        self.temperature = config["temperature"]
        self.max_tokens = config["max_tokens"]

    def generate(self, prompt: str) -> str:
        response = self.client.messages.create(
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text


class GeminiModel:
    def __init__(self, model_name: str = config["gemini_model"]) -> None:
        api_key = os.getenv(config["gemini_api_key_env"])
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY is not set in the environment.")
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)

    def generate(self, prompt: str) -> str:
        response = self.model.generate_content(prompt)
        return (getattr(response, "text", "") or "").strip()