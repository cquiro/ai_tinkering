from openai import OpenAI
import os

api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    print("Error: OPENAI_API_KEY not set in environment")
    exit(1)

client = OpenAI(api_key=api_key)

try:
    question = input("Ask me about personal finances: ")
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": question}],
        max_tokens=300
    )

    print("Bot Response:")
    print(response.choices[0].message.content)
    
except client.error.RateLimitError as e:
    if "insuficient_quota" in str(e):
        print("Error: You've exceeded your OpenAI API quota. Please check your plan and billing details at platform.openai.com/account/billing.")
        print("Steps to fix:")
        print("1. Add more credits or upgrade to a paid plan.")
        print("2. Regenerate your API key after adding credits.")
        print("3. Wait 10-15 minutes and retry.")
    else:
        print(f"Rate limit error: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
