from openai import OpenAI
import os
from datetime import datetime

api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    print("Error: OPENAI_API_KEY not set in environment")
    exit(1)

client = OpenAI(api_key=api_key)

log_file = "finance_bot_history.txt"

print("Welcome to your Personal Finance Bot!")

with open(log_file, "a") as f:
    f.write(f"\n--- Session started at {datetime.now()} ---\n")

    while True:
        question = input("Ask me about personal finances (or type 'exit' to quit): ")
        if question.lower() == "exit":
            print("Goodbye!")
            f.write("\n--- Session ended ---\n")
            break

        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": question}],
                max_tokens=300
            )

            answer = response.choices[0].message.content
            print("Bot Response:")
            print(answer)
            print()

            f.write(f"Q: {question}\nA: {answer}\n\n")
        
        except client.error.RateLimitError as e:
            if "insuficient_quota" in str(e):
                print("Error: You've exceeded your OpenAI API quota. Please check your plan and billing details at platform.openai.com/account/billing.")
                print("Steps to fix:")
                print("1. Add more credits or upgrade to a paid plan.")
                print("2. Regenerate your API key after adding credits.")
                print("3. Wait 10-15 minutes and retry.")
            else:
                print(f"Rate limit error: {e}")
            f.write("--- Session ended due to error ---\n")
            break
        except Exception as e:
            print(f"Unexpected error: {e}")
            f.write("--- Session ended due to error ---\n")
            break
