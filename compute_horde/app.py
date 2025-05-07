from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import uvicorn
from vllm_llm import ReproducibleVLLM
from schema import ChatRequest
from loguru import logger
import os

MODEL_PATH = os.getenv('MODEL_PATH')

class LlmModelApp:
    def __init__(self):
        self.llm = ReproducibleVLLM(model_id=MODEL_PATH)
        self.app = FastAPI()
        self.app.post("/generate")(self.generate)
        self.app.post("/generate_logits")(self.generate_logits)

    async def generate(self, request: ChatRequest):
        try:
            sampling_parameters = request.sampling_parameters.dict()
            del sampling_parameters["top_n"]
            result = await self.llm.generate(
                messages=[m.dict() for m in request.messages],
                seed=request.seed,
                sampling_params=sampling_parameters,
                continue_last_message=request.continue_last_message
            )
            return {
                "result": result
            }
        except Exception as e:
            return JSONResponse(status_code=500, content={"error": str(e)})

    async def generate_logits(self, request: ChatRequest):
        try:
            sampling_parameters = request.sampling_parameters.dict()
            del sampling_parameters["max_new_tokens"]
            top_n = sampling_parameters.pop("top_n", 10)
            logits, prompt = await self.llm.generate_logits(
                messages=[m.dict() for m in request.messages],
                top_n=top_n,
                seed=request.seed,
                sampling_params=sampling_parameters,
                continue_last_message=request.continue_last_message
            )
            return {
                "logits": logits,
                "prompt": prompt
            }
        except Exception as e:
            return JSONResponse(status_code=500, content={"error": str(e)})

    def run(self):
        uvicorn.run(self.app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    server = LlmModelApp()
    server.run()
