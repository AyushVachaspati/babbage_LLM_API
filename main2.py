import uvicorn
from transformers import AutoModelForCausalLM, AutoTokenizer
from pydantic import BaseModel
from fastapi import FastAPI

model_path = "./models/santacoder"
device = "cpu" # "cuda" for GPU usage or "cpu" for CPU usage
print("Loading Model")
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path,trust_remote_code=True).to(device)
app = FastAPI()
print("Model Loaded")

class Data(BaseModel):
    prompt: str

@app.post("/code_complete_test")
async def complete_code(prompt:Data):
    print(prompt.prompt)
    inputs = tokenizer.encode(prompt.prompt, return_tensors="pt").to(device)
    output = model.generate(inputs,min_length=10,max_length=25)
    result = tokenizer.decode(output[0])
    result = result.removeprefix(prompt.prompt)
    print(prompt.prompt+result)
    return {"result": result}

if __name__ == "__main__":
    print("Starting Server")
    uvicorn.run("main:app", port=8000, log_level="info")
    print("Server UP!!")


