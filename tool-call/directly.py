from transformers import AutoTokenizer, AutoModelForCausalLM
from trl.trainer.utils import SIMPLE_CHAT_TEMPLATE
import json

MODEL = "meta-llama/Llama-3.1-8B-Instruct"

# First, define a tool
def lcm(a: int, b: int) -> int:
    """
    Calculate the least common multiple of two integers.
    
    Args:
        a: The first integer.
        b: The second integer.
    Returns:
        The least common multiple of the two integers.
    """
    def gcd(x: int, y: int) -> int:
        while y:
            x, y = y, x % y
        return x

    return abs(a * b) // gcd(a, b)

# Next, create a chat and apply the chat template
messages = [
  {"role": "system", "content": "You are a bot that responds to queries."},
  {"role": "user", "content": "What is LCM of 127491203981242758992 and 1327812793127893129283?"}
]

tokenizer = AutoTokenizer.from_pretrained(MODEL, padding_side="left")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(MODEL, device_map="auto")
if tokenizer.chat_template is None:
    tokenizer.chat_template = SIMPLE_CHAT_TEMPLATE

inputs = tokenizer.apply_chat_template(messages, tools=[lcm], add_generation_prompt=True, return_tensors="pt").to(model.device)

outputs = model.generate(inputs, max_new_tokens=64)
response = tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
print(response)

# Extract the function call and arguments from the response
tool_call = json.loads(response)
function_name = tool_call["name"]
arguments = tool_call["parameters"]

print(f"Function to call: {function_name} with arguments {arguments}")
if function_name == "lcm":
    a = arguments["a"]
    b = arguments["b"]
    result = lcm(a, b)
    print(f"The least common multiple of {a} and {b} is {result}")

    inputs_with_tool_response = tokenizer.apply_chat_template(
        messages + [
            {"role": "assistant", "tool_calls": [{"type": "function", "function": {"name": function_name, "arguments": arguments}}]},
            {"role": "tool", "name": function_name, "content": str(result)}
        ],
        tools=[lcm],
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)
    outputs_with_tool_response = model.generate(inputs_with_tool_response, max_new_tokens=64)
    final_response = tokenizer.decode(outputs_with_tool_response[0], skip_special_tokens=True)
    print("Final response from the model:")
    print(final_response)