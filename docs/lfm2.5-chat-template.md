# LFM2.5 Chat Template Reference

Source: <https://huggingface.co/LiquidAI/LFM2.5-1.2B-Instruct>

## Special Tokens

| Token ID | Token | Description |
|----------|-------|-------------|
| 0 | `<\|pad\|>` | Padding token |
| 1 | `<\|startoftext\|>` | Start of conversation |
| 2 | `<\|endoftext\|>` | End of text |
| 3 | `<\|fim_pre\|>` | Fill-in-middle prefix |
| 4 | `<\|fim_mid\|>` | Fill-in-middle middle |
| 5 | `<\|fim_suf\|>` | Fill-in-middle suffix |
| 6 | `<\|im_start\|>` | Start of message (followed by role) |
| 7 | `<\|im_end\|>` | End of message |
| 8 | `<\|tool_list_start\|>` | Start of tool list definition |
| 9 | `<\|tool_list_end\|>` | End of tool list definition |
| 10 | `<\|tool_call_start\|>` | Start of function call block |
| 11 | `<\|tool_call_end\|>` | End of function call block |
| 12 | `<\|tool_response_start\|>` | Start of tool response |
| 13 | `<\|tool_response_end\|>` | End of tool response |

## Basic Chat Template

LFM2.5 uses a **ChatML-like format**:

```
<|startoftext|><|im_start|>system
You are a helpful assistant trained by Liquid AI.<|im_end|>
<|im_start|>user
What is C. elegans?<|im_end|>
<|im_start|>assistant
```

### Roles

- `system` - System instructions
- `user` - User message
- `assistant` - Model response
- `tool` - Tool/function execution result

## Tool Use Template

### 1. Function Definition (in system prompt)

Provide tools as a JSON array in the system prompt:

```
<|startoftext|><|im_start|>system
List of tools: [{"name": "get_candidate_status", "description": "Retrieves the current status of a candidate in the recruitment process", "parameters": {"type": "object", "properties": {"candidate_id": {"type": "string", "description": "Unique identifier for the candidate"}}, "required": ["candidate_id"]}}]<|im_end|>
```

### 2. Function Call (assistant response)

LFM2.5 writes **Pythonic function calls** by default:

```
<|im_start|>assistant
<|tool_call_start|>[get_candidate_status(candidate_id="12345")]<|tool_call_end|>Checking the current status of candidate ID 12345.<|im_end|>
```

**Note**: You can request JSON function calls by specifying it in the system prompt.

### 3. Function Result (tool role)

```
<|im_start|>tool
[{"candidate_id": "12345", "status": "Interview Scheduled", "position": "Clinical Research Associate", "date": "2023-11-20"}]<|im_end|>
```

### 4. Final Answer

```
<|im_start|>assistant
The candidate with ID 12345 is currently in the "Interview Scheduled" stage for the position of Clinical Research Associate, with an interview date set for 2023-11-20.<|im_end|>
```

## Complete Tool Use Example

```
<|startoftext|><|im_start|>system
List of tools: [{"name": "get_candidate_status", "description": "Retrieves the current status of a candidate in the recruitment process", "parameters": {"type": "object", "properties": {"candidate_id": {"type": "string", "description": "Unique identifier for the candidate"}}, "required": ["candidate_id"]}}]<|im_end|>
<|im_start|>user
What is the current status of candidate ID 12345?<|im_end|>
<|im_start|>assistant
<|tool_call_start|>[get_candidate_status(candidate_id="12345")]<|tool_call_end|>Checking the current status of candidate ID 12345.<|im_end|>
<|im_start|>tool
[{"candidate_id": "12345", "status": "Interview Scheduled", "position": "Clinical Research Associate", "date": "2023-11-20"}]<|im_end|>
<|im_start|>assistant
The candidate with ID 12345 is currently in the "Interview Scheduled" stage for the position of Clinical Research Associate, with an interview date set for 2023-11-20.<|im_end|>
```

## Using with Transformers

### Basic Chat

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "LiquidAI/LFM2.5-1.2B-Instruct"
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", dtype="bfloat16")
tokenizer = AutoTokenizer.from_pretrained(model_id)

prompt = "What is C. elegans?"

input_ids = tokenizer.apply_chat_template(
    [{"role": "user", "content": prompt}],
    add_generation_prompt=True,
    return_tensors="pt",
    tokenize=True,
).to(model.device)

output = model.generate(
    input_ids,
    do_sample=True,
    temperature=0.1,
    top_k=50,
    top_p=0.1,
    repetition_penalty=1.05,
    max_new_tokens=512,
)
```

### With Tools

```python
tools = [
    {
        "name": "get_weather",
        "description": "Get current weather for a location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string", "description": "City name"}
            },
            "required": ["location"]
        }
    }
]

messages = [{"role": "user", "content": "What's the weather in Tokyo?"}]

input_ids = tokenizer.apply_chat_template(
    messages,
    tools=tools,
    add_generation_prompt=True,
    return_tensors="pt",
    tokenize=True,
).to(model.device)
```

## Model Specifications

- **Context length**: 32,768 tokens
- **Vocabulary size**: 65,536
- **Knowledge cutoff**: Mid-2024
- **Languages**: English, Arabic, Chinese, French, German, Japanese, Korean, Spanish

### Recommended Generation Parameters

```python
temperature=0.1
top_k=50
top_p=0.1
repetition_penalty=1.05
```

## References

- [Chat Template Documentation](https://docs.liquid.ai/lfm/key-concepts/chat-template)
- [Tool Use Documentation](https://docs.liquid.ai/lfm/key-concepts/tool-use)
- [HuggingFace Model Page](https://huggingface.co/LiquidAI/LFM2.5-1.2B-Instruct)
