from cegsr.tasks.qa import QATask
from cegsr.trajectories.schema import ExperienceNode, TaskSample


def test_mcq_accuracy_accepts_label_and_text_in_sentence():
    task = QATask()
    sample = TaskSample(
        sample_id="mcq1",
        question="Where would you keep a rug near your front door?",
        answer="D. living room",
        choices=["A. persia", "B. desk", "C. table", "D. living room", "E. hall"],
        task_type="mmlu_style",
    )
    metrics = task.evaluate_prediction(sample, "The correct answer is D. living room.")
    assert metrics["accuracy"] == 1
    assert metrics["mcq_accuracy"] == 1
    assert metrics["exact_match"] == 0


def test_mcq_accuracy_accepts_sentence_with_choice_text_only():
    task = QATask()
    sample = TaskSample(
        sample_id="mcq2",
        question="Danny found an old film in a sealed what?",
        answer="D. cabinet",
        choices=["A. clingfilm", "B. disneyland", "C. cave", "D. cabinet", "E. movie"],
        task_type="mmlu_style",
    )
    metrics = task.evaluate_prediction(sample, "Danny found an old film in a sealed cabinet.")
    assert metrics["accuracy"] == 1
    assert metrics["mcq_accuracy"] == 1


def test_build_prompt_includes_mcq_format_instruction_for_summarizer():
    task = QATask()
    sample = TaskSample(
        sample_id="mcq3",
        question="The earth is one planet in what?",
        answer="C. solar system",
        choices=["A. tree", "B. orbit", "C. solar system", "D. photograph", "E. dreams"],
        task_type="mmlu_style",
    )
    messages = task.build_prompt(
        sample=sample,
        role="summarizer",
        retrieved_experience=[],
        history=[],
        system_prompt="system",
        extra_context={},
    )
    assert "Final Answer: <LETTER>. <choice text>" in messages[1]["content"]


def test_build_prompt_includes_repair_guidance_and_retrieval_caution():
    task = QATask()
    sample = TaskSample(
        sample_id="repair1",
        question="Which option is correct?",
        answer="A. first",
        choices=["A. first", "B. second"],
        task_type="qa",
    )
    messages = task.build_prompt(
        sample=sample,
        role="solver",
        retrieved_experience=[
            ExperienceNode(
                node_id="n1",
                text="Answer: A. first because it matches the stem.",
                role="solver",
                task_type="qa",
                credit=0.91,
                source_episode_id="ep1",
                source_turn_ids=["t1"],
                is_repaired=True,
            )
        ],
        history=[],
        system_prompt="system",
        extra_context={
            "repair_mode": True,
            "repair_reason": "low credit on solver turn",
            "preserved_context": ["Planner already narrowed the answer space to A/B."],
        },
    )
    content = messages[1]["content"]
    assert "Selective repair mode" in content
    assert "Repair target: low credit on solver turn" in content
    assert "Preserved high-credit context" in content
    assert "reuse a snippet only if it clearly matches this question" in content
