"""Local test helper for OpenAI-compatible chat completion."""

from __future__ import annotations

import os


def build_client():
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("未安装 openai 依赖") from exc

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("请先设置 OPENAI_API_KEY 环境变量")

    base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    return OpenAI(api_key=api_key, base_url=base_url)


class GPTClient:
    def __init__(self, model: str = "gpt-4o-mini") -> None:
        self.model = model
        self.client = build_client()

    def chat(
        self,
        text: str,
        temperature: float = 0.8,
        system: str = "你是人工智能助手",
        top_p: float = 1.0,
    ) -> str:
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": text},
            ],
            temperature=temperature,
            top_p=top_p,
            stream=False,
        )
        return completion.choices[0].message.content or ""


if __name__ == "__main__":
    client = GPTClient()
    print(client.chat("Hello, how are you?"))
