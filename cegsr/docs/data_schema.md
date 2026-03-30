# Data Schema

## TaskSample
- `sample_id`
- `question`
- `answer`
- `context`
- `choices`
- `task_type`
- `metadata`

## AgentTurn
- `turn_id`
- `role`
- `prompt_messages`
- `response`
- `dependencies`
- `citations`
- `latency_s`
- `input_tokens`
- `output_tokens`
- `meta`

## SubTrajectory
- `sub_id`
- `turn_ids`
- `roles`
- `summary`
- `start_turn`
- `end_turn`

## EpisodeTrajectory
- `episode_id`
- `sample`
- `turns`
- `subtrajectories`
- `final_prediction`
- `metrics`
- `reward`
- `credit_records`
- `repair_records`

## CreditRecord
- `target_type`
- `target_id`
- `total`
- `signals`
- `details`

## RepairRecord
- `repair_id`
- `target_type`
- `target_id`
- `old_span`
- `new_span`
- `why_repaired`
- `kept_context_turn_ids`
- `verifier_before`
- `verifier_after`

## ExperienceNode / ExperienceEdge
Used by the causal experience graph.
