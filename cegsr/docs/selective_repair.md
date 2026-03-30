# Selective Repair

Current implementation:
1. detect low-credit target,
2. preserve high-credit prefix context,
3. rerun only the local target + downstream suffix,
4. keep before/after versions in `RepairRecord`.

This is intentionally a practical v1.
It can later be upgraded to:
- learned detector,
- graph-aware patching,
- partial DAG replay instead of ordered suffix replay,
- verifier-conditioned acceptance thresholds.
