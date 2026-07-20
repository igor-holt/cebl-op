# Eulerian Codex Entry — CEBL-Op

- **Name**: CEBL-Op (Continuous Energy-Based Logic — Operational)
- **Class**: Reasoning Verification Primitive
- **Origin**: TORAX Logic-Transport Pilot → v3.0 (February 2026, audited May 2026)
- **Owner agent**: Genesis Conductor (Evidence plane)
- **Interfaces**: HAVIS v2.0, DVLO step 5→6 gate, diamondNode audit endpoint
- **Key invariants**:
  - T = σ(u) — no hard clip, no probability claim
  - K strictly upper-triangular
  - SHATTERED = convergence ∧ energy ∧ max-violation composite
  - IFT is a screening signal only; cold snap is ground truth
- **Worked example**: Yennefer of Vengerberg constraint graph (4-node, DUCTILE_COLLAPSE, truth_drop=0.77)
- **Status**: SPEC (not yet deployed)
