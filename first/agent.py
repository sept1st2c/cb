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


def planner_agent(user_input, memory):
    system_prompt = f"""
You are a PLANNER AI.

Your job is to break the user's request into a sequence of actions.

KNOWN USER INFO:
{memory["user_profile"]}

AVAILABLE ACTIONS:
- remember (key, value)
- get_current_time
- calculate (input)
- final

RULES:
- Output ONLY ONE JSON object.
- Output a JSON with a "plan" array.
- Each step must be ONE action.
- Do NOT answer the user.
- Do NOT execute tools.
- End every plan with a "final" action.


Plan Schema:
{{
  "plan": [
    {{ "action": "remember", "key": "name/age/bheaviour traits/etc", "value": "user_name_here/user_age_here/user_behaviour_traits_here/etc" }},
    {{ "action": "get_current_time" }},
    {{ "action": "calculate", "input": "2+2" }},
    {{ "action": "final" }}
  ]
}}

"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input}
        ]
    )

    return json.loads(response.choices[0].message.content)



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
    memory["tool_outputs"] = {}

    plan_obj = planner_agent(user_input, memory)
    print("🗺️ PLAN:", plan_obj)

    for step in plan_obj["plan"]:
        action = step["action"]

        # ---- REMEMBER ----
        if action == "remember":
            memory["user_profile"][step["key"]] = step["value"]

        # ---- TIME TOOL ----
        elif action == "get_current_time":
            memory["tool_outputs"]["current_time"] = get_current_time()

        # ---- CALCULATOR ----
        elif action == "calculate":
            memory["tool_outputs"]["calculation"] = calculate(step["input"])

        # ---- FINAL ----
        elif action == "final":
            save_memory(memory)
            return finalize_answer(user_input, memory)


def finalize_answer(user_input, memory):
    system_prompt = """
You are an AI assistant answering the user.

Use:
- user profile memory
- tool outputs

Answer the FULL question clearly.
"""
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": f"""
User question:
{user_input}

User profile:
{memory["user_profile"]}

Tool outputs:
{memory["tool_outputs"]}
"""
            }
        ]
    )

    return response.choices[0].message.content


# ---------- RUN ----------
while True:
    user_input = input("\nYou: ")
    if user_input.lower() == "exit":
        break

    reply = run_agent(user_input)
    print("🤖 Agent:", reply)
