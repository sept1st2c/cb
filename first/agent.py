import os
import json
from openai import OpenAI
from dotenv import load_dotenv
from PIL import Image
import pytesseract

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


def ocr_extract_text(image_path):
    try:
        image = Image.open(image_path)
        text = pytesseract.image_to_string(image)
        return text.strip()
    except Exception as e:
        return f"OCR failed: {str(e)}"



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
- ocr_extract_text (input: image_path)
    => Use ocr_extract_text when:
        - the user uploads an image
        - the user asks to read, analyze, or extract text from an image
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
    {{ "action": "ocr_extract_text", "input": "image_path_given"}}
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


# ---------- AGENT LOOP ----------
def run_agent(user_input):
    
    memory = load_memory()
    memory.setdefault("user_profile", {})
    memory["tool_outputs"] = {}
    memory.setdefault("documents", [])


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
        
        elif action == "ocr_extract_text":
            extracted_text = ocr_extract_text(step["input"])
            memory["tool_outputs"]["ocr_text"] = extracted_text

            memory["documents"].append({
                "source": step["input"],
                "text": extracted_text
            })



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

Documents involved before:
{memory["documents"]}
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
