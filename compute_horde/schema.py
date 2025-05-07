from pydantic import BaseModel, Field
from typing import List, Optional, Literal

class ChatMessage(BaseModel):
    content: str
    role: Literal["user", "assistant", "system"]

class SamplingParameters(BaseModel):
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.95
    top_k: Optional[int] = 50
    top_n: Optional[int] = 10
    max_new_tokens: Optional[int] = 1024
    presence_penalty: Optional[float] = 0.0
    frequency_penalty: Optional[float] = 0.0


class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    seed: Optional[int] = 42
    sampling_parameters: Optional[SamplingParameters] = SamplingParameters()
    continue_last_message: Optional[bool] = False