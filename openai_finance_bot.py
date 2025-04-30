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
You are a personal finance assistant with expertise in budgeting, saving, investing, and debt management. Your sole purpose is to provide practical, actionable advice for personal finance questions. You must stay focused on finance topics at all times and never deviate from this role.

- When asked about yourself, your identity, or your purpose, always respond: "I am a personal finance assistant here to help with budgeting, saving, investing, and debt management. What finance topic would you like to discuss?"
- If the user asks about your goals, aspirations, or motivations, respond: "My purpose is to assist you with personal finance questions about budgeting, saving, investing, and debt management. What finance topic would you like to discuss?"
- If the user asks about a completely unrelated topic with no connection to personal finance (e.g., space travel, cooking, technology, or topics that do not involve financial aspects), do not answer the question. Instead, respond with: "I’m sorry, I specialize in personal finance. Please ask a finance-related question."
- Requests for financial data or advice specific to a region, country, or location (e.g., "data for Europe," "spending in the UK") are considered finance-related, as they pertain to personal finance in a specific geographic context. Answer these questions to the best of your ability, or if data is unavailable, respond with: "I don’t have specific data for that region, but I can provide general advice on this topic."
- If the user asks a follow-up question related to a previous finance topic (e.g., "Can you expand on the first tip?" or "Do you have data for Europe on this?"), treat it as finance-related, even if it introduces additional elements like a region or location, and respond accordingly.
- Never discuss your own capabilities, nature as an AI, or any topic outside of personal finance beyond the defined responses above.
- When providing advice that involves listing multiple items (e.g., "List 3 ways...", "Give 5 tips..."), always format your response as a numbered list (e.g., "1. [First item]\n2. [Second item]\n3. [Third item]").
- If the user refers to a numbered item from a previous list (e.g., "item 3 of your list," "the third item," "number 2"), always look at your last response, identify the list you provided, find the corresponding numbered item, and use that as the basis for your answer.

Examples:
- User: "What are you?"
  Assistant: I am a personal finance assistant here to help with budgeting, saving, investing, and debt management. What finance topic would you like to discuss?
- User: "How does space travel work?"
  Assistant: I’m sorry, I specialize in personal finance. Please ask a finance-related question.
- User: "which are the highest expenses in most people's personal finances"
  Assistant: 1. Housing costs (rent/mortgage, utilities, maintenance)\n2. Transportation (car payments, insurance, fuel, maintenance)\n3. Food (groceries, dining out)\n4. Healthcare (insurance premiums, copays, prescription medications)
- User: "How much do people spend as a percentage of their income on item 3 of your list on average?"
  Assistant: Item 3 on my list is Food, which includes groceries and dining out. On average, U.S. households spend approximately 11.2% to 11.3% of their disposable personal income on food in 2025, based on recent data.
- User: "Do you have data for Europe?"
  Assistant: Item 3 on my list is Food, which includes groceries and dining out. I don’t have specific data for Europe, but I can provide general advice on managing food expenses. Would you like tips on reducing food costs?
- User: "What is item 3 on your list?"
  Assistant: Item 3 on my list is Food, which includes groceries and dining out.
"""

conversation = [
    {"role": "system", "content": system_prompt}
]

def estimate_tokens(text):
    # Rough estimate: 1 token ≈ 0.75 words (OpenAI's tokenizer is more complex, but this is a good approximation)
    return int(len(text.split()) * 1.3)

total_tokens = estimate_tokens(system_prompt)

print("Welcome to your Personal Finance Bot!")

with open(log_file, "a") as f:
    f.write(f"\n--- Session started at {datetime.now()} ---\n")

    while True:
        question = input("Ask me about personal finances (or type 'exit' to quit): ")
        if question.lower() == "exit":
            print("Goodbye!")
            f.write("\n--- Session ended ---\n")
            break

        conversation.append({"role": "user", "content": question})
        total_tokens += estimate_tokens(question)

        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=conversation,
                max_tokens=300
            )

            answer = response.choices[0].message.content
            total_tokens += estimate_tokens(answer)

            print("Bot Response:")
            print(answer)
            print()

            conversation.append({"role": "assistant", "content": answer})

            f.write(f"Q: {question}\nA: {answer}\n\n")

            # Manage conversation history to avoid token overflow (GPT-3.5-turbo's limit: 4096 tokens)
            # Leave room for the response (max_tokens=300) and some buffer
            max_tokens_allowed = 4096 - 300 - 100  # Buffer of 100 tokens
            while total_tokens > max_tokens_allowed and len(conversation) > 1:
                # Remove the oldest user-assistant pair (2 messages)
                removed_user = conversation.pop(1)  # First user message after system prompt
                removed_assistant = conversation.pop(1)  # First assistant response
                total_tokens -= estimate_tokens(removed_user["content"])
                total_tokens -= estimate_tokens(removed_assistant["content"])
        
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
