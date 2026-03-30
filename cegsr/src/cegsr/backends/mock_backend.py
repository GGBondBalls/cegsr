from __future__ import annotations

import ast
import operator as op
import re
from typing import Any

from cegsr.backends.base import BaseBackend, BackendResponse, GenerationConfig


_ALLOWED = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.truediv,
    ast.FloorDiv: op.floordiv,
    ast.Pow: op.pow,
    ast.USub: op.neg,
}


def safe_eval(expr: str) -> str:
    """Safely evaluate a tiny arithmetic expression."""
    def _eval(node: ast.AST) -> float:
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED:
            return _ALLOWED[type(node.op)](_eval(node.left), _eval(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED:
            return _ALLOWED[type(node.op)](_eval(node.operand))
        raise ValueError("unsupported expression")

    tree = ast.parse(expr, mode="eval")
    value = _eval(tree.body)
    if abs(value - int(value)) < 1e-9:
        return str(int(value))
    return f"{value:.4f}".rstrip("0").rstrip(".")


def extract_question_text(messages: list[dict[str, str]]) -> str:
    text = "\n".join(m.get("content", "") for m in messages)
    match = re.search(r"Question:\s*(.+?)(?:\n\n|$)", text, re.S)
    return match.group(1).strip() if match else text


def extract_arithmetic_answer(text: str) -> str | None:
    candidates = re.findall(r"([-+()\d/*\s]{3,})", text)
    candidates = sorted(candidates, key=len, reverse=True)
    for candidate in candidates:
        cleaned = candidate.strip()
        if re.fullmatch(r"[\d\s\-+/*().]+", cleaned) and any(ch.isdigit() for ch in cleaned):
            try:
                return safe_eval(cleaned)
            except Exception:
                continue
    return None


class MockBackend(BaseBackend):
    """Deterministic backend for smoke tests and unit tests."""

    backend_name = "mock"

    def generate(
        self,
        messages: list[dict[str, str]],
        generation_config: GenerationConfig | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> BackendResponse:
        metadata = metadata or {}
        role = metadata.get("role", "solver")
        prompt = "\n".join(m.get("content", "") for m in messages)
        question = extract_question_text(messages)
        answer = extract_arithmetic_answer(question) or metadata.get("gold_answer", "42")

        if role == "planner":
            text = f"Plan: compute the core quantity carefully, then pass candidate answer {answer} forward."
        elif role == "solver":
            text = f"Reasoning: solving `{question}` gives {answer}. Proposed answer: {answer}"
        elif role == "verifier":
            if answer in prompt:
                text = "VERDICT: correct\nScore: 0.95\nIssue: none"
            else:
                text = f"VERDICT: likely incorrect\nScore: 0.25\nIssue: recompute, expected around {answer}"
        elif role == "summarizer":
            text = f"Final Answer: {answer}"
        else:
            text = f"Answer: {answer}"
        return BackendResponse(text=text, raw={"mock": True}, input_tokens=self.count_tokens(messages), output_tokens=len(text.split()))
