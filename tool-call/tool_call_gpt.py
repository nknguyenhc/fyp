from openai import OpenAI
import json
import dotenv

dotenv.load_dotenv()

client = OpenAI()

# 1. Define a list of callable tools for the model
tools = [
    {
        "type": "function",
        "name": "lcm",
        "description": "Calculate the least common multiple of two integers.",
        "parameters": {
            "type": "object",
            "properties": {
                "a": {
                    "type": "integer",
                    "description": "The first integer",
                },
                "b": {
                    "type": "integer",
                    "description": "The second integer",
                },
            },
            "required": ["a", "b"],
        },
    },
]

def lcm(a, b):
    """Calculate the least common multiple of two integers."""
    def gcd(x, y):
        while y:
            x, y = y, x % y
        return x

    return abs(a * b) // gcd(a, b)

# Create a running input list we will add to over time
input_list = [
    {"role": "user", "content": "What is LCM of 128729731289738912783 and 2812793712987398127983712 ?"}
]

# 2. Prompt the model with tools defined
response = client.responses.create(
    model="gpt-5",
    tools=tools,
    input=input_list,
)

# Save function call outputs for subsequent requests
input_list += response.output

for item in response.output:
    if item.type == "function_call":
        if item.name == "lcm":
            # 3. Execute the function logic for lcm
            args = json.loads(item.arguments)
            result = lcm(args["a"], args["b"])
            
            # 4. Provide function call results to the model
            input_list.append({
                "type": "function_call_output",
                "call_id": item.call_id,
                "output": json.dumps({
                  "result": result
                })
            })

print("Final input:")
print(input_list)

response = client.responses.create(
    model="gpt-5",
    tools=tools,
    input=input_list,
)

# 5. The model should be able to give a response!
print("Final output:")
print(response.model_dump_json(indent=2))
print("\n" + response.output_text)