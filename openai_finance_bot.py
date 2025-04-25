from openai import OpenAI
import os
from datetime import datetime

api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    print("Error: OPENAI_API_KEY not set in environment")
    exit(1)

client = OpenAI(api_key=api_key)

log_file = "finance_bot_history.txt"

system_prompt = """
You are a personal finance assistant with expertise in budgeting, saving, investing, and debt management. Your sole purpose is to provide practical, actionable advice for personal finance questions. You must stay focused on finance topics at all times.

- When asked about yourself, always respond: "I am a personal finance assistant here to help with budgeting, saving, investing, and debt management. What finance topic would you like to discuss?"
- If the user asks about an unrelated topic (e.g., space travel, cooking, technology, or anything not directly related to personal finance), do not answer the question. Instead, respond with: "I’m sorry, I specialize in personal finance. Please ask a finance-related question."
- If the user asks a follow-up question (e.g., "Can you expand on the first tip?"), ensure your response remains finance-related by referring to the previous finance context.
- Never discuss your own capabilities, goals, or nature as an AI beyond the defined role above.

Examples:
- User: "What are you?"
  Assistant: I am a personal finance assistant here to help with budgeting, saving, investing, and debt management. What finance topic would you like to discuss?
- User: "Which is your goal?"
  Assistant: I am a personal finance assistant here to help with budgeting, saving, investing, and debt management. My purpose is to assist you with finance questions. What finance topic would you like to discuss?
- User: "How does space travel work?"
  Assistant: I’m sorry, I specialize in personal finance. Please ask a finance-related question.
- User: "What’s the best recipe for lasagna?"
  Assistant: I’m sorry, I specialize in personal finance. Please ask a finance-related question.
- User: "List 3 ways to save for a vacation on a $40k salary."
  Assistant: [Provide finance-related answer]
- User: "Can you expand on the first tip?"
  Assistant: [Provide finance-related expansion based on prior context]
"""

conversation = [
    {"role": "system", "content": system_prompt}
]

def is_obviously_off_topic(question):
    off_topic_keywords = [
        "space travel", "rocket", "astronaut", "planet", "cooking", "recipe",
        "technology", "software", "hardware", "lasagna", "medical", "health"
    ]
    return any(keyword in question.lower() for keyword in off_topic_keywords)

def is_finance_related(text):
    finance_keywords = [
        "budget", "save", "saving", "invest", "debt", "income", "expense", "salary",
        "money", "finance", "financial", "vacation fund", "emergency fund", "retirement",
        "mortgage", "rent", "utilities", "groceries", "transportation", "spending"
    ]
    return any(keyword in text.lower() for keyword in finance_keywords)


print("Welcome to your Personal Finance Bot!")

with open(log_file, "a") as f:
    f.write(f"\n--- Session started at {datetime.now()} ---\n")

    while True:
        question = input("Ask me about personal finances (or type 'exit' to quit): ")
        if question.lower() == "exit":
            print("Goodbye!")
            f.write("\n--- Session ended ---\n")
            break

        if is_obviously_off_topic(question):
            answer = "I’m sorry, I specialize in personal finance. Please ask a finance-related question."
            print("Bot Response:")
            print(answer)
            print()
            f.write(f"Q: {question}\nA: {answer}\n\n")
            continue

        conversation.append({"role": "user", "content": question})

        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": question}],
                max_tokens=300
            )

            answer = response.choices[0].message.content

            # if not is_finance_related(question) and not is_finance_related(answer):
            #     answer = "I’m sorry, I specialize in personal finance. Please ask a finance-related question."

            print("Bot Response:")
            print(answer)
            print()

            conversation.append({"role": "assistant", "content": answer})

            f.write(f"Q: {question}\nA: {answer}\n\n")

            # Manage conversation history to avoid token overflow
            if len(conversation) > 7:  # System prompt + 3 exchanges
                conversation = [conversation[0]] + conversation[-6:]
        
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
