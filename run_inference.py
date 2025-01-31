import os
import openai
import together
from dotenv import load_dotenv

load_dotenv()

# Load API keys from environment
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY", "")

openai_client = openai.Client(
    api_key=OPENAI_API_KEY,
    # Optionally configure other parameters like:
    # max_retries=3,
    # request_timeout=15,
)


def query_openai(system_prompt: str, prompt: str, model: str = "gpt-4o-mini") -> str:
    """
    Query an OpenAI Chat model using the new openai>=1.0.0 client.
    """
    try:
        response = openai_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            temperature=0.7,
            max_tokens=150,
        )
        # The "content" is in response.choices[0].message["content"]
        return response.choices[0].message["content"].strip()
    except Exception as e:
        print("Error querying OpenAI:", e)
        return ""

def query_together(prompt: str, model: str = "togethercomputer/RedPajama-INCITE-7B-Chat") -> str:
    """Query Together's text generation API."""
    if not TOGETHER_API_KEY:
        raise ValueError("Together API key is missing. Please set TOGETHER_API_KEY.")

    try:
        # Using `together.llm.complete` for a text completion
        # Model name examples: "togethercomputer/RedPajama-INCITE-Chat-3B-v1",
        #                     "togethercomputer/RedPajama-INCITE-Instruct-3B-v1", etc.
        response = together.llm.complete(
            model_name=model,
            prompt=prompt,
            max_tokens=150,
            temperature=0.7,
        )
        # 'response' is typically a dictionary with a 'choices' list, similar to OpenAI.
        # The content might vary based on the Together LLM.
        if response and "choices" in response and len(response["choices"]) > 0:
            return response["choices"][0].get("text", "").strip()
        return ""
    except Exception as e:
        print("Error querying Together:", e)
        return ""

def main():
    prompt = "What is the capital of France?"

    print("\n--- Querying OpenAI ---")
    openai_answer = query_openai(system_prompt="Answer the question below", prompt=prompt)
    print("OpenAI answer:", openai_answer)

    # print("\n--- Querying Together ---")
    # together_answer = query_together(prompt)
    # print("Together answer:", together_answer)

if __name__ == "__main__":
    main()