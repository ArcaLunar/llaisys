import os
import time
from dataclasses import dataclass
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import AutoTokenizer

from ..libllaisys import DeviceType
from ..models.qwen2 import Qwen2


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = Field(default="qwen2")
    messages: list[ChatMessage]
    max_tokens: int = Field(default=128, ge=1, le=4096)
    temperature: float = Field(default=0.8, gt=0.0)
    top_p: float = Field(default=0.9, gt=0.0, le=1.0)
    top_k: int = Field(default=40, ge=0)
    stream: bool = False


class ChoiceMessage(BaseModel):
    role: Literal["assistant"] = "assistant"
    content: str


class ChatCompletionChoice(BaseModel):
    index: int
    message: ChoiceMessage
    finish_reason: Literal["stop", "length"]


class ChatCompletionUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(BaseModel):
    id: str
    object: Literal["chat.completion"] = "chat.completion"
    created: int
    model: str
    choices: list[ChatCompletionChoice]
    usage: ChatCompletionUsage


@dataclass
class ChatRuntime:
    tokenizer: AutoTokenizer
    model: Qwen2


def _render_prompt(messages: list[ChatMessage], tokenizer: AutoTokenizer) -> str:
    role_map = [{"role": m.role, "content": m.content} for m in messages]
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            role_map,
            tokenize=False,
            add_generation_prompt=True,
        )

    chunks = []
    for message in role_map:
        chunks.append(f"<{message['role']}>\n{message['content']}\n")
    chunks.append("<assistant>\n")
    return "".join(chunks)


def _decode_new_text(tokenizer: AutoTokenizer, all_tokens: list[int], prompt_len: int) -> str:
    return tokenizer.decode(all_tokens[prompt_len:], skip_special_tokens=True)


def create_app(runtime: ChatRuntime) -> FastAPI:
    app = FastAPI(title="LLAISYS Chat Server", version="0.1.0")

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
    def chat_completions(req: ChatCompletionRequest) -> ChatCompletionResponse:
        if req.stream:
            raise HTTPException(status_code=400, detail="stream=true is not supported yet")
        if not req.messages:
            raise HTTPException(status_code=400, detail="messages must not be empty")

        prompt = _render_prompt(req.messages, runtime.tokenizer)
        prompt_tokens = runtime.tokenizer.encode(prompt)

        generated_tokens = runtime.model.generate(
            prompt_tokens,
            max_new_tokens=req.max_tokens,
            top_k=req.top_k,
            top_p=req.top_p,
            temperature=req.temperature,
        )

        answer_text = _decode_new_text(runtime.tokenizer, generated_tokens, len(prompt_tokens))
        completion_tokens = max(0, len(generated_tokens) - len(prompt_tokens))

        return ChatCompletionResponse(
            id=f"chatcmpl-{int(time.time() * 1000)}",
            created=int(time.time()),
            model=req.model,
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=ChoiceMessage(content=answer_text),
                    finish_reason="stop",
                )
            ],
            usage=ChatCompletionUsage(
                prompt_tokens=len(prompt_tokens),
                completion_tokens=completion_tokens,
                total_tokens=len(prompt_tokens) + completion_tokens,
            ),
        )

    return app


def _parse_device(device_name: str) -> DeviceType:
    if device_name.lower() == "nvidia":
        return DeviceType.NVIDIA
    return DeviceType.CPU


def build_runtime_from_env() -> ChatRuntime:
    model_path = os.environ.get("LLAISYS_MODEL_PATH", "./data")
    device = _parse_device(os.environ.get("LLAISYS_DEVICE", "cpu"))

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = Qwen2(model_path=model_path, device=device)
    return ChatRuntime(tokenizer=tokenizer, model=model)
