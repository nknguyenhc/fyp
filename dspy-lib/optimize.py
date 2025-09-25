import dspy
lm = dspy.LM("openai/gpt-4o-mini", api_key="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")
dspy.configure(lm=lm)

## React agent
from dspy.datasets import HotPotQA

def search_wikipedia(query: str) -> list[str]:
    results = dspy.ColBERTv2(url="http://20.102.90.50:2017/wiki17_abstracts")(query, k=3)
    return [x["text"] for x in results]

trainset = [x.with_inputs('question') for x in HotPotQA(train_seed=2024, train_size=500).train]
react = dspy.ReAct("question -> answer", tools=[search_wikipedia])

tp = dspy.MIPROv2(metric=dspy.evaluate.answer_exact_match, auto="light", num_threads=24)
optimized_react = tp.compile(react, trainset=trainset, requires_permission_to_run=False)
print(optimized_react)

### Result
# react = Predict(StringSignature(question, trajectory -> next_thought, next_tool_name, next_tool_args
#     instructions="Given the fields `question`, produce the fields `answer`.\n\nYou are an Agent. In each episode, you will be given the fields `question` as input. And you can see your past trajectory 
# so far.\nYour goal is to use one or more of the supplied tools to collect any necessary information for producing `answer`.\n\nTo do this, you will interleave next_thought, next_tool_name, and next_tool_args in each turn, and also when finishing the task.\nAfter each tool call, you receive a resulting observation, which gets appended to your trajectory.\n\nWhen writing next_thought, you may reason about the current situation and plan for future steps.\nWhen selecting the next_tool_name and its next_tool_args, the tool must be one of:\n\n(1) search_wikipedia. It takes arguments {'query': {'type': 
# 'string'}}.\n(2) finish, whose description is <desc>Marks the task as complete. That is, signals that all information for producing the outputs, i.e. `answer`, are now available to be extracted.</desc>. It takes arguments {}.\nWhen providing `next_tool_args`, the value inside the field must be in JSON format"
#     question = Field(annotation=str required=True json_schema_extra={'__dspy_field_type': 'input', 'prefix': 'Question:', 'desc': '${question}'})
#     trajectory = Field(annotation=str required=True json_schema_extra={'__dspy_field_type': 'input', 'prefix': 'Trajectory:', 'desc': '${trajectory}'})
#     next_thought = Field(annotation=str required=True json_schema_extra={'__dspy_field_type': 'output', 'prefix': 'Next Thought:', 'desc': '${next_thought}'})
#     next_tool_name = Field(annotation=Literal['search_wikipedia', 'finish'] required=True json_schema_extra={'__dspy_field_type': 'output', 'prefix': 'Next Tool Name:', 'desc': '${next_tool_name}'})   
#     next_tool_args = Field(annotation=dict[str, Any] required=True json_schema_extra={'__dspy_field_type': 'output', 'prefix': 'Next Tool Args:', 'desc': '${next_tool_args}'})
# ))
# extract.predict = Predict(StringSignature(question, trajectory -> reasoning, answer
#     instructions='Given the fields `question`, produce the fields `answer`.'
#     question = Field(annotation=str required=True json_schema_extra={'__dspy_field_type': 'input', 'prefix': 'Question:', 'desc': '${question}'})
#     trajectory = Field(annotation=str required=True json_schema_extra={'__dspy_field_type': 'input', 'prefix': 'Trajectory:', 'desc': '${trajectory}'})
#     reasoning = Field(annotation=str required=True json_schema_extra={'prefix': "Reasoning: Let's think step by step in order to", 'desc': '${reasoning}', '__dspy_field_type': 'output'})
#     answer = Field(annotation=str required=True json_schema_extra={'__dspy_field_type': 'output', 'prefix': 'Answer:', 'desc': '${answer}'})
# ))

## Classification
# from typing import Literal

# # Define the DSPy module for classification. It will use the hint at training time, if available.
# signature = dspy.Signature("text, hint -> label").with_updated_fields("label", type_=Literal[tuple(CLASSES)])
# classify = dspy.ChainOfThought(signature)

# # Optimize via BootstrapFinetune.
# optimizer = dspy.BootstrapFinetune(metric=(lambda x, y, trace=None: x.label == y.label), num_threads=24)
# optimized = optimizer.compile(classify, trainset=trainset)

# optimized(text="What does a pending cash withdrawal mean?")

# # For a complete fine-tuning tutorial, see: https://dspy.ai/tutorials/classification_finetuning/
