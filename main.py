import copy
import uvicorn
from fastapi import FastAPI
from llama_cpp import Llama
from pydantic import BaseModel

llm = Llama(model_path="./models/ggml-vicuna-13B-1.1-q4_0.bin")
app = FastAPI()


@app.get("/")
async def hello():
    return {"hello": "world"}


@app.get('/code_complete')
async def model():
    prompt = "def hello_world():\n"
    stream = llm(prompt,
                 max_tokens=100,
                 stop=["\n"],
                 echo=True)
    result = copy.deepcopy(stream)['choices'][0]['text']
    result = result.removeprefix(prompt)
    return {"result": result}

class Data(BaseModel):
    prompt: str

@app.post("/code_complete_test")
async def complete_code(prompt:Data):
    print(prompt.prompt)
    stream = llm(prompt.prompt,
                 max_tokens=100,
                 stop=["\n"],
                 echo=True)
    result = copy.deepcopy(stream)['choices'][0]['text']
    result = result.removeprefix(prompt.prompt)
    print(prompt.prompt+result)
    return {"result": result}


if __name__ == "__main__":
    uvicorn.run("main:app", port=8000, log_level="info")

