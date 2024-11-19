from openai import OpenAI
from openai.types.embedding import Embedding

class OpenAIClient:
    def __init__(self, api_key: str):
        self.client = OpenAI(
            api_key=api_key
        )

    def chat_completion(self, messages: list[dict], model: str = "gpt-4o-mini") -> str:
        completion = self.client.chat.completions.create(
            model=model,
            messages=messages
        )

        return completion.choices[0].message.content
    
    def embed_texts(self, texts: list[str], model: str = "text-embedding-3-small") -> list[Embedding]:
        return self.client.embeddings.create(
            model=model,
            input=texts
        ).data
