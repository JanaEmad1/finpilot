"""HTTP API.  Run:  uvicorn finpilot.api:app --reload

Demo shortcut: the request carries `user_id`. In a real bank the user id would come from
the authenticated session (e.g. a verified JWT), never from the request body.
"""
from __future__ import annotations

import logging
import time
from functools import lru_cache

from fastapi import FastAPI
from pydantic import BaseModel, Field

from finpilot import llm
from finpilot.agent import Agent
from finpilot.db import get_engine
from finpilot.intent.predict import load_intent_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("finpilot.api")

app = FastAPI(title="FinPilot", description="LLM banking assistant (demo with synthetic data)")


class ChatRequest(BaseModel):
    user_id: int = Field(..., ge=1, examples=[1])
    message: str = Field(..., min_length=1, max_length=1000, examples=["How much did I spend on groceries last month?"])
    session_id: str = Field("default", max_length=64)


@lru_cache(maxsize=1)
def get_agent() -> Agent:
    return Agent(get_engine(), load_intent_model())


@app.get("/health")
def health() -> dict:
    agent = get_agent()
    return {"status": "ok", "intent_model": agent.intent_model.name,
            "handoff_threshold": agent.threshold, "llm": "gemini" if llm.available() else "template-only"}


@app.post("/chat")
def chat(request: ChatRequest) -> dict:
    start = time.perf_counter()
    reply = get_agent().chat(request.user_id, request.message, request.session_id)
    latency_ms = round((time.perf_counter() - start) * 1000)
    # log decisions and latency, but never the raw message (it may contain personal data)
    log.info("chat route=%s intent=%s conf=%s llm=%s redacted=%s latency_ms=%s",
             reply.route, reply.intent, reply.confidence, reply.used_llm, reply.redacted, latency_ms)
    return {**reply.to_dict(), "latency_ms": latency_ms}
