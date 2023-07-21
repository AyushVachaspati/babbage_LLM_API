import uvicorn
from transformers import AutoModelForCausalLM, AutoTokenizer
from pydantic import BaseModel
from fastapi import FastAPI
import torch

model_path = "./models/santacoder"
checkpoint = "bigcode/santacoder"
app = FastAPI()
device = "cuda"  # "cuda" for GPU usage or "cpu" for CPU usage

cache = {}
tokenizer = None
model = None

class Data(BaseModel):
    prompt: str

@app.on_event('startup')
def load_model():
    global tokenizer
    global model
    print("Loading Model")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint,cache_dir=model_path)
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.bos_token
    model = AutoModelForCausalLM.from_pretrained(checkpoint,cache_dir=model_path,trust_remote_code=True,torch_dtype=torch.float16).to(device)
    # model = AutoModelForCausalLM.from_pretrained(checkpoint,cache_dir=model_path,trust_remote_code=True).to(device)
    # model = AutoModelForCausalLM.from_pretrained(model_path,trust_remote_code=True,load_in_8bit=True)
    print("Model Loaded")


@app.post("/code_complete_test")
async def complete_code(prompt:Data):
    global tokenizer
    global model
    prompt = prompt.prompt
    if not prompt:
        # print("Empty Prompt")
        return {"result":""}
    if prompt in cache:
        # print(cache[prompt])
        return {"result": cache[prompt]}
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    output = model.generate(**inputs,pad_token_id=tokenizer.eos_token_id,min_new_tokens=1,max_new_tokens=25)
    result = tokenizer.decode(output[0]).rstrip(tokenizer.convert_ids_to_tokens(tokenizer.eos_token_id))

    # print(result)
    cache[prompt] = result
    return {"result": result}

if __name__ == "__main__":
    print("Starting Server")
    uvicorn.run("main:app",host='0.0.0.0', port=8000, log_level="debug")
