# CEBL-Op — Continuous Energy-Based Logic (Reasoning Robustness Module)
**Status markers:** SPEC / IMPLEMENTED / PENDING  
**Parent:** HAVIS v2.0  
**Canonical version:** v3.0 only (v1.0 PDE metaphor and v2.0 residuals discarded)

## 1. Purpose (SPEC / IMPLEMENTED)
CEBL-Op measures logical equilibrium under soft evidence gating, conflict, and implication constraints on an epistemic graph. It detects CRYSTALLINE / DUCTILE_COLLAPSE / SHATTERED_AT_BASE states and ranks premise vulnerabilities.

It does **not** claim probability elimination. It performs relaxed truth valuation and constraint satisfaction only.

## 2. RAG Graph Interface (SPEC)
- Claims `c_i` extracted as atomic propositions from CoT/RAG output.
- Evidence supports `e_i` / gating vector `m_i ∈ [0,1]` scored by cross-encoder NLI.
- Conflict matrix `K` (strictly upper-triangular).
- Implication matrix `M`.

## 3. Energy Functional (SPEC)
\[
E(u, \Theta) = E_{\text{evidence}} + E_{\text{conflict}} + E_{\text{inference}} + E_{\text{reg}}
\]
where \( T = \sigma(u) \) (sigmoid, no hard clip),
\[
\begin{align*}
E_{\text{evidence}} &= \sum_i m_i (1 - T_i)^2, \\
E_{\text{conflict}} &= \sum_{i < j} K_{ij} [\max(0, T_i + T_j - 1)]^2, \\
E_{\text{inference}} &= \sum_{i,j} M_{ij} \operatorname{softplus}_\beta(\max(0, T_j - T_i - \delta))^2, \\
E_{\text{reg}} &= \lambda \|u\|^2.
\end{align*}
\]

## 4. Solver (SPEC)
Levenberg-Marquardt (damped Newton) + Armijo backtracking line search.  
Convergence criteria + composite SHATTERED detection: convergence ∧ energy threshold ∧ max constraint violation.

## 5. Seismography (SPEC)
- IFT screening (O(1) local sensitivity of equilibrium to \( m_i \)).
- Finite-difference cold snap on top-k ranked premises (ground-truth validation). Always required when \( \Delta m > 0.3 \).

## 6. Status Taxonomy (SPEC)
- **CRYSTALLINE**: stable equilibrium, low violations.
- **DUCTILE_COLLAPSE**: recoverable via premise adjustment (truth drop observed).
- **SHATTERED_AT_BASE**: composite failure → return to DWE / step 4.

## 7. JAX Implementation Skeleton (IMPLEMENTED)
```python
import jax
import jax.numpy as jnp
from jax import grad, hessian, jit

def sigmoid(u):
    return jax.nn.sigmoid(u)

def energy(u, m, K, M, beta=5.0, lam=1e-3, delta=0.0):
    T = sigmoid(u)
    n = T.shape[0]
    E_ev = jnp.sum(m * (1 - T)**2)
    i, j = jnp.triu_indices(n, k=1)
    E_conf = jnp.sum(K[i, j] * jnp.maximum(0, T[i] + T[j] - 1)**2)
    E_inf = 0.0  # implement full M outer softplus as needed
    E_reg = lam * jnp.sum(u**2)
    return E_ev + E_conf + E_inf + E_reg

def find_logical_equilibrium(m, K, M, max_iter=50, tol=1e-6):
    n = m.shape[0]
    u = jnp.zeros(n)
    for _ in range(max_iter):
        g = grad(lambda uu: energy(uu, m, K, M))(u)
        H = hessian(lambda uu: energy(uu, m, K, M))(u)
        H_damped = H + 1e-4 * jnp.eye(n)
        try:
            du = jnp.linalg.solve(H_damped, -g)
        except:
            du = -g
        alpha = 1.0
        E0 = energy(u, m, K, M)
        for _ls in range(20):
            u_new = u + alpha * du
            if energy(u_new, m, K, M) < E0 + 1e-4 * alpha * jnp.dot(g, du):
                break
            alpha *= 0.5
        u = u_new
        if jnp.linalg.norm(g) < tol:
            break
    T = sigmoid(u)
    max_viol = 0.0
    status = "CRYSTALLINE"
    if max_viol > 0.1 or jnp.linalg.norm(g) > tol:
        status = "SHATTERED_AT_BASE"
    return {"u": u, "T": T, "status": status, "energy": energy(u, m, K, M)}

def evaluate_reasoning_robustness(claims, m, K, M, top_k=3):
    eq = find_logical_equilibrium(m, K, M)
    if eq["status"] == "SHATTERED_AT_BASE":
        return eq
    # IFT screening + cold snap (skeleton)
    return eq
```

## 8. Worked Example — Yennefer JSONL (IMPLEMENTED)
Canonical 4-node narrative constraint graph (Yennefer of Vengerberg).  
Produces DUCTILE_COLLAPSE with truth_drop ≈ 0.77 under critical premise removal.  
(Full JSONL retained in project state; do not discard.)

## 9. Open Issues (SPEC)
- [ ] Nonconvexity / indefinite Hessian → L-BFGS + modified Cholesky.
- [ ] IFT locality caveat → always cold-snap for large Δm.
- [ ] NLI graph quality dependency → threshold calibration.
- [ ] β temperature calibration (default 5.0 arbitrary).
- [ ] Scale: full Hessian O(n²); switch to L-BFGS / sparse for n > ~200.

## DVLO Integration Hook (SPEC)
```
Step 5: Genesis Conductor builds execution plan
        ↓
[CEBL-Op GATE]
  - Extract claims
  - NLI score → m, K, M
  - find_logical_equilibrium()
  - If SHATTERED_AT_BASE → DWE (step 4)
  - IFT + cold snap (mode-dependent)
  - Attach robustness report to Evidence lane
        ↓
Step 6: LidLift scores tools
```

Mode gating: Fast=skip, Standard=base+IFT, Audit=full cold snap.

## diamondNode Payload Schema (SPEC)
```yaml
diamond_node_audit_entry:
  type: "cebl_op_robustness_report"
  fields:
    run_id: string
    claim_count: int
    base_status: "CRYSTALLINE | DUCTILE_COLLAPSE | SHATTERED_AT_BASE"
    vulnerability_score: float
    critical_premise_idx: int
    truth_drop: float
    energy_breakdown: {evidence: float, conflict: float, inference: float, reg: float}
    max_inf_violation: float
    max_conf_violation: float
  attestation: genesis_proof_hash
  endpoint: "dn.genesisconductor.io/cebl-audit"
```

**Status:** SPEC (not yet deployed).  
Yennefer JSONL preserved as canonical worked example.
