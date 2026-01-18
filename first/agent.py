import os
import json
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=os.getenv("GROQ_API_KEY")
)


MEMORY_FILE = "memory.json"

def load_memory():
    if not os.path.exists(MEMORY_FILE):
        return {}
    with open(MEMORY_FILE, "r") as f:
        return json.load(f)

def save_memory(memory):
    with open(MEMORY_FILE, "w") as f:
        json.dump(memory, f, indent=2)


# ---------- TOOL ----------
def get_current_time():
    return "Current time is 10:30 PM"

def calculate(expression):
    try:
        return str(eval(expression))
    except Exception:
        return "Invalid calculation"


# ---------- AGENT BRAIN ----------
def agent_think(user_input, memory):
    system_prompt = f"""
You are a simple AI agent.

MEMORY:
{memory}



TOOLS AVAILABLE:

1. get_current_time
- Use when the user asks for the current time or date.

2. calculate
- Use when the user asks to calculate or compute something.
- Input must be a valid math expression (example: "2 + 3 * 4")

Decide the next best action to move toward answering the full question.


RESPONSE FORMAT:
You MUST respond with VALID JSON ONLY.
No extra text.
No markdown.
No explanations.

IMPORTANT RULES:
- You must output EXACTLY ONE JSON object.
- You must choose ONLY ONE action per response.
- NEVER output multiple JSON objects.
- NEVER plan multiple steps in one response.
- The system will call you again after each action.



Use a tool:
{{
  "action": "get_current_time"
}}

{{
  "action": "calculate",
  "input": "math expression"
}}

Remember information:
{{
  "action": "remember",
  "key": "name/age/bheaviour traits/etc",
  "value": "user_name_here/user_age_here/user_behaviour_traits_here/etc"
}}

Finish answering:
{{
  "action": "final",
  "input": "full answer to the user's question"
}}
IMPORTANT: Do not apologize. Do not say you are an AI. If asked for the time, use the tool.
"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input}
        ]
    )

    return response.choices[0].message.content

# ---------- AGENT LOOP ----------
def run_agent(user_input):
    
    memory = load_memory()
    memory.setdefault("user_profile", {})
    memory.setdefault("tool_outputs", {})
    memory["user_input"] = user_input

    while True:
        decision = agent_think(user_input, memory)
        print("🧠 Agent decision:", decision)
        data = json.loads(decision)

        action = data["action"]

        # ---- REMEMBER ----
        if action == "remember":
            memory["user_profile"][data["key"]] = data["value"]
            save_memory(memory)
            user_input = f"""
Original question:
{user_input}

things i remembered so far about the user:
{memory["user_profile"]}
"""
            continue

        # ---- TOOL ----
        if action == "get_current_time":
            result = get_current_time()
            memory["tool_outputs"]["current_time"] = result

            # IMPORTANT:
            # After tool use, we let the agent think AGAIN
            user_input = f"""
Original question:
{user_input}

Tool results so far:
{memory["tool_outputs"]}
"""
            
            continue
        
        if action == "calculate":
            result = calculate(data["input"])
            memory["tool_outputs"]["calculation"] = result

            user_input = f"""
Original question:
{user_input}

Tool results so far:
{memory["tool_outputs"]}
"""
            continue

        # ---- FINAL ANSWER ----
        if action == "final":
            save_memory(memory)
            return data["input"]


# ---------- RUN ----------
while True:
    user_input = input("\nYou: ")
    if user_input.lower() == "exit":
        break

    reply = run_agent(user_input)
    print("🤖 Agent:", reply)
