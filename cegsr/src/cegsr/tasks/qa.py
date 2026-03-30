from __future__ import annotations

import re
from typing import Any

from cegsr.tasks.base import BaseTask, normalize_text
from cegsr.trajectories.schema import AgentTurn, TaskSample


class QATask(BaseTask):
    task_type = "qa"

    _CHOICE_RE = re.compile(r"^\s*([A-Z])[.)]\s*(.+?)\s*$")
    _LABEL_PATTERNS = (
        re.compile(r"(?:final answer|correct answer|best answer|proposed answer|answer|option|choice)\s*(?:is|:)?\s*([A-Z])\b", re.I),
        re.compile(r"^\s*([A-Z])[.)]\s*", re.I),
        re.compile(r"\(([A-Z])\)", re.I),
    )

    def _parse_choices(self, sample: TaskSample) -> dict[str, str]:
        parsed: dict[str, str] = {}
        for choice in sample.choices:
            match = self._CHOICE_RE.match(choice.strip())
            if not match:
                continue
            parsed[match.group(1).upper()] = normalize_text(match.group(2))
        return parsed

    def _parse_gold_choice(self, sample: TaskSample) -> tuple[str | None, str]:
        gold = sample.answer.strip()
        match = self._CHOICE_RE.match(gold)
        if match:
            return match.group(1).upper(), normalize_text(match.group(2))
        return None, normalize_text(gold)

    def _extract_mcq_choice(self, text: str, sample: TaskSample) -> tuple[str | None, str | None]:
        choice_map = self._parse_choices(sample)
        if not choice_map:
            return None, None

        normalized = normalize_text(text)
        for pattern in self._LABEL_PATTERNS:
            match = pattern.search(text)
            if match:
                label = match.group(1).upper()
                if label in choice_map:
                    return label, choice_map[label]

        text_matches = [label for label, choice_text in choice_map.items() if choice_text and choice_text in normalized]
        if len(text_matches) == 1:
            label = text_matches[0]
            return label, choice_map[label]

        if normalized in choice_map.values():
            for label, choice_text in choice_map.items():
                if choice_text == normalized:
                    return label, choice_text

        return None, None

    def _format_instructions(self, sample: TaskSample, role: str) -> str:
        if not sample.choices:
            if role == "summarizer":
                return "Output exactly one line in the format `Final Answer: <answer>`."
            if role == "solver":
                return "Put the candidate answer on the first line in the format `Answer: <answer>`."
            if role == "verifier":
                return "Output at least two lines: `VERDICT: correct|incorrect` and `Score: <0-1>`."
            return ""

        if role == "solver":
            return (
                "This is a multiple-choice question. Use exactly one listed option. "
                "Write the first line exactly as `Answer: <LETTER>. <choice text>`, then optionally add brief reasoning."
            )
        if role == "summarizer":
            return (
                "This is a multiple-choice question. Output exactly one line and nothing else: "
                "`Final Answer: <LETTER>. <choice text>`."
            )
        if role == "verifier":
            return (
                "This is a multiple-choice question. Verify whether the selected option is one of the listed choices and whether the reasoning supports it. "
                "Output lines `VERDICT: correct|incorrect` and `Score: <0-1>`. Be conservative when uncertain."
            )
        if role == "planner":
            return "This is a multiple-choice question. Restrict analysis to the listed options and do not invent new answers."
        return ""

    def build_prompt(
        self,
        sample: TaskSample,
        role: str,
        retrieved_experience: list[Any],
        history: list[AgentTurn],
        system_prompt: str,
        extra_context: dict[str, Any],
    ) -> list[dict[str, str]]:
        messages: list[dict[str, str]] = [{"role": "system", "content": system_prompt}]
        snippets = []
        guidance = []
        if extra_context.get("repair_mode"):
            guidance.append(
                "Selective repair mode: preserve validated earlier reasoning and rewrite only the local span most likely causing the error."
            )
            repair_reason = extra_context.get("repair_reason")
            if repair_reason:
                guidance.append(f"Repair target: {repair_reason}")
            preserved_context = [str(item).strip() for item in extra_context.get("preserved_context", []) if str(item).strip()]
            if preserved_context:
                guidance.append("Preserved high-credit context:")
                guidance.extend(f"- {item[:240]}" for item in preserved_context)
        elif extra_context.get("rewrite_entire_trajectory"):
            guidance.append(
                "Retry mode: the previous full trajectory failed. Produce a fresh, corrected trajectory instead of copying the earlier answer."
            )
        if retrieved_experience:
            snippets.append(
                "Retrieved experience (high-credit hints only; reuse a snippet only if it clearly matches this question, choices, and current role):"
            )
            for node in retrieved_experience:
                prefix = "repaired" if getattr(node, "is_repaired", False) else "prior"
                snippets.append(
                    f"- [{node.role}|credit={node.credit:.2f}|{prefix}] {node.text[:160]}"
                )
        if sample.metadata.get("previous_failure"):
            snippets.append("Previous failed trajectory to learn from:")
            snippets.append(str(sample.metadata["previous_failure"])[:800])
        history_block = []
        if history:
            history_block.append("History:")
            for turn in history:
                history_block.append(f"[{turn.turn_id}|{turn.role}] {turn.response}")
        choices_block = ""
        if sample.choices:
            choices_block = "\nChoices:\n" + "\n".join(f"- {c}" for c in sample.choices)
        format_instructions = self._format_instructions(sample, role)
        user_message = "\n\n".join(
            [
                f"Question: {sample.question}{choices_block}",
                f"Context: {sample.context or '(none)'}",
                format_instructions,
                "\n".join(guidance).strip(),
                "\n".join(snippets).strip(),
                "\n".join(history_block).strip(),
                f"Current role: {role}",
            ]
        ).strip()
        messages.append({"role": "user", "content": user_message})
        return messages

    def extract_prediction(self, text: str) -> str:
        patterns = [
            r"Final Answer:\s*(.+)",
            r"Proposed answer:\s*(.+)",
            r"Answer:\s*(.+)",
            r"The correct answer is\s*(.+)",
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.I)
            if match:
                return match.group(1).strip().splitlines()[0]
        return text.strip().splitlines()[-1].strip()

    def evaluate_prediction(self, sample: TaskSample, prediction: str) -> dict[str, Any]:
        pred = normalize_text(prediction)
        gold = normalize_text(sample.answer)
        exact = int(pred == gold)
        mcq = 0
        if sample.choices:
            gold_label, gold_text = self._parse_gold_choice(sample)
            pred_label, pred_text = self._extract_mcq_choice(prediction, sample)
            mcq = int(
                (pred_label is not None and pred_label == gold_label)
                or (pred_text is not None and pred_text == gold_text)
                or pred == gold
            )
        accuracy = mcq if sample.choices else exact
        return {"accuracy": accuracy, "exact_match": exact, "mcq_accuracy": mcq}
