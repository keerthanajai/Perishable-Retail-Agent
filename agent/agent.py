import os
import anthropic
from dotenv import load_dotenv
from agent.memory import ConversationMemory
from agent.tools import TOOL_DEFINITIONS, execute_tool

load_dotenv()

client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

SYSTEM_PROMPT = """You are a Demand Intelligence Agent for a perishable retail business in Ecuador.
You help store managers and executives make smart inventory decisions by forecasting demand,
explaining what drives it, and recommending actions.

You have access to 4 tools:
- forecast_demand: predicts unit sales for a store + item
- explain_forecast: explains WHY using SHAP feature importance
- query_sales_history: gets recent historical context
- recommend_action: gives a full forecast + explanation + inventory recommendation

IMPORTANT RULES:
1. Always use tools to get real data — never guess numbers.
2. For complex questions, chain tools: get forecast → explain → recommend.
3. If the user asks a follow-up like "why?" or "now do store 5", use context from earlier in the conversation.
4. Keep responses concise and business-focused. Translate technical outputs into plain English.
5. If you don't have enough info (missing store_id or item_id), ask the user for it.
6. Valid store IDs are 1-54. Here are example store+item pairs with strong sales history 
   you can use to demo: (51, 1239986), (44, 1503844), (3, 1503844), (44, 1473474), (11, 584126).

You are talking to non-technical retail managers. Be clear, confident, and actionable.
"""


class DemandAgent:
    def __init__(self):
        self.memory = ConversationMemory(max_turns=20)

    def chat(self, user_message: str) -> str:
        """
        Main entry point. Takes a user message, runs the ReAct loop,
        and returns the final response.
        """
        # Add user message to memory
        self.memory.add_user(user_message)

        # Build messages for Claude
        messages = self.memory.get_history()[:-1]  # history without latest
        messages.append({"role": "user", "content": user_message})

        # ReAct loop — keep going until Claude stops using tools
        max_iterations = 5
        iteration = 0

        while iteration < max_iterations:
            iteration += 1

            response = client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=1024,
                system=SYSTEM_PROMPT,
                tools=TOOL_DEFINITIONS,
                messages=messages
            )

            # Check stop reason
            if response.stop_reason == "end_turn":
                final_text = self._extract_text(response)
                self.memory.add_assistant(final_text)
                return final_text

            elif response.stop_reason == "tool_use":
                messages.append({
                    "role": "assistant",
                    "content": response.content
                })

                # Execute all requested tools
                tool_results = []
                for block in response.content:
                    if block.type == "tool_use":
                        print(f"  🔧 Calling tool: {block.name} with {block.input}")

                        # Update memory context for follow-ups
                        self.memory.update_context(**block.input)

                        result = execute_tool(block.name, block.input)
                        print(f"Tool result: {result[:100]}...")

                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": result
                        })

                messages.append({
                    "role": "user",
                    "content": tool_results
                })

            else:
                break

        fallback = "I wasn't able to complete the analysis. Please try rephrasing your question."
        self.memory.add_assistant(fallback)
        return fallback

    def _extract_text(self, response) -> str:
        """Extracts plain text from Claude's response."""
        texts = [block.text for block in response.content if hasattr(block, "text")]
        return " ".join(texts).strip()

    def reset(self):
        """Clears conversation history."""
        self.memory.clear()
        print("Conversation reset.")



if __name__ == "__main__":
    print("=" * 55)
    print("  Demand Intelligence Agent — CLI")
    print("  Type 'exit' to quit | 'reset' to clear history")
    print("=" * 55)

    agent = DemandAgent()

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() == "exit":
            print("Goodbye!")
            break
        if user_input.lower() == "reset":
            agent.reset()
            continue

        print("\nAgent: ", end="", flush=True)
        response = agent.chat(user_input)
        print(response)