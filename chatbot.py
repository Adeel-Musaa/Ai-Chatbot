import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_huggingface import HuggingFacePipeline
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda

model_id = "Qwen/Qwen2-0.5B-Instruct"

print(f"Loading model: {model_id}")

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Load model
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype="auto",
    device_map="auto"
)

# Create Hugging Face pipeline
pipe = pipeline(
    "text-generation",
    model=model,                 # <-- Missing tha
    tokenizer=tokenizer,
    max_new_tokens=128,          # <-- max_new_token nahi, max_new_tokens
    temperature=0.7,
    do_sample=True
)

# LangChain wrapper
llm = HuggingFacePipeline(pipeline=pipe)


# Prompt function
def create_qwen_prompt(user_input: str, history: list = None):
    if history is None:
        history = []

    messages = [
        {
            "role": "system",
            "content": "You are a helpful AI assistant."
        }
    ]

    messages.extend(history)
    messages.append(
        {
            "role": "user",
            "content": user_input
        }
    )

    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )


print("Setup Complete")