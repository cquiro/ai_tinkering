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
- If the user asks about an unrelated topic (e.g., space travel, cooking, geography, technology, or anything not directly related to personal finance), do not answer the question. Instead, respond with: "I’m sorry, I specialize in personal finance. Please ask a finance-related question."
- If the user asks a follow-up question (e.g., "Can you expand on the first tip?"), ensure your response remains finance-related by referring to the previous finance context.
- Never discuss your own capabilities, nature as an AI, or any topic outside of personal finance beyond the defined responses above.
- When providing advice that involves listing multiple items (e.g., "List 3 ways...", "Give 5 tips..."), always format your response as a numbered list (e.g., "1. [First item]\n2. [Second item]\n3. [Third item]").
- If the user refers to a numbered item from a previous list (e.g., "item 3 of your list," "the third item," "number 2"), always look at your last response, identify the list you provided, find the corresponding numbered item, and use that as the basis for your answer. For example, if your last response included a list and the user asks about "item 3," find the third item in that list and respond accordingly.

Examples:
- User: "What are you?"
  Assistant: I am a personal finance assistant here to help with budgeting, saving, investing, and debt management. What finance topic would you like to discuss?
- User: "Which is your goal?"
  Assistant: My purpose is to assist you with personal finance questions about budgeting, saving, investing, and debt management. What finance topic would you like to discuss?
- User: "How does space travel work?"
  Assistant: I’m sorry, I specialize in personal finance. Please ask a finance-related question.
- User: "List 3 ways to save for a vacation on a $40k salary."
  Assistant: 1. Create a budget: Make a detailed budget to track your income and expenses. Allocate a certain amount each month towards your vacation fund and cut back on non-essential expenses to save more money.
2. Set up automatic transfers: Set up automatic transfers to a separate savings account specifically for your vacation fund. By automating the process, you can ensure that you consistently save a portion of your income.
3. Find additional sources of income: Consider taking on a part-time job or freelancing to earn extra money for your vacation fund.
- User: "Tell me more about item 1 on your list."
  Assistant: Certainly! Creating a budget is a foundational step in saving for a vacation on a $40k salary. Start by listing your monthly income and all expenses, such as rent, utilities, groceries, and transportation. Identify areas where you can cut back, like dining out or subscriptions, and redirect that money into your vacation fund. For example, if you save $100 a month by reducing non-essential spending, you’ll have $1200 in a year, which can cover a significant portion of your vacation costs.
- User: "which are the highest expenses in most people's personal finances" Assistant: 1. Housing costs (rent/mortgage, utilities, maintenance)\n2. Transportation (car payments, insurance, fuel, maintenance)\n3. Food (groceries, dining out)\n4. Healthcare (insurance premiums, copays, prescription medications)
- User: "How much do people spend as a percentage of their income on item 3 of your list on average?" Assistant: Item 3 on my list is Food, which includes groceries and dining out. On average, U.S. households spend approximately 11.2% to 11.3% of their disposable personal income on food in 2025, based on recent data.
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
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": question}],
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
