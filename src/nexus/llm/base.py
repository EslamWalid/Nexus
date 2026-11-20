import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

from nexus.config import (
    LLM_MODEL,
    MAX_RETRIES,
    MAX_TOKEN,
    THINKING_LEVEL,
    TIMEOUT,
)

ROOT_DIR = Path.cwd()

load_dotenv(ROOT_DIR / ".env", override=True)


class GeminiLLM:
    def __init__(self):

        api_key1 = os.getenv("GEMINI_API_KEY_LLM1")
        api_key2 = os.getenv("GEMINI_API_KEY_LLM2")

        if not api_key1:
            raise ValueError("GEMINI_API_KEY_LLM1 is missing")

        self.primary_llm = ChatGoogleGenerativeAI(
            model=LLM_MODEL,
            api_key=api_key1,
            max_tokens=MAX_TOKEN,
            timeout=TIMEOUT,
            max_retries=MAX_RETRIES,
            thinking_level=THINKING_LEVEL,
        )

        self.llm = self.primary_llm

        if api_key2:
            self.fallback_llm = ChatGoogleGenerativeAI(
                model=LLM_MODEL,
                api_key=api_key2,
                max_tokens=MAX_TOKEN,
                timeout=TIMEOUT,
                max_retries=MAX_RETRIES,
                thinking_level=THINKING_LEVEL,
            )

            self.llm = self.primary_llm.with_fallbacks([self.fallback_llm])

    async def ainvoke(self, prompt):
        """Generate a response asynchronously."""
        return await self.llm.ainvoke(prompt)


if __name__ == "__main__":
    import asyncio

    async def main():
        llm = GeminiLLM()
        response = await llm.ainvoke(
            "What are some good Italian restaurants near the Eiffel Tower in Paris?"
        )
        print(response)

    asyncio.run(main())
