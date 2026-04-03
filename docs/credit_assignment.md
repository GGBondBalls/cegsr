# Credit Assignment

Implemented signal families:

1. Outcome credit
- reward-linked heuristic signal
- slightly favors later decisive turns

2. Verifier credit
- parse explicit verifier score when present
- otherwise back off to local consistency heuristics

3. Dependency credit
- boosts turns reused by later turns

Fusion:
- weighted average over signal families
- preserves intermediate decomposition in `CreditRecord.signals`
