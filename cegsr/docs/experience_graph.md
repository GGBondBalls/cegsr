# Experience Graph Design

Nodes can come from:
- high-credit turns
- high-credit subtrajectories
- repaired spans

Edge types:
- `temporal`
- `support`
- `contradiction`
- `same_role_pattern`
- `repaired_from`

Retrieval:
1. embed current role + task + query + recent history,
2. filter by task type and role neighborhood,
3. top-k semantic retrieval,
4. optional one-hop neighborhood expansion.
