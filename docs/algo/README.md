# Algorithm Reference Index

## Complete Algorithm Taxonomy

This directory contains comprehensive documentation for all elevator dispatching algorithms implemented in the smart-lift-RL system.

---

## 1. **Deterministic Heuristics**

### [SCAN (Directional Collective Control)](SCAN.md)
- **Type**: Mechanical sweeping algorithm
- **Complexity**: O(n) per tick
- **Status**: ✅ Production-ready
- **Best for**: Directional traffic (morning/evening rush)
- **Mean wait**: ~4–8 seconds
- **Key insight**: Industry standard for 99% of elevators worldwide

### [Nearest First (Greedy Search)](NEAREST_FIRST.md)
- **Type**: Naive greedy assignment
- **Complexity**: O(n²) per tick
- **Status**: ❌ Broken (included as negative example)
- **Mean wait**: ~34 seconds (average) but **320+ seconds** (outliers)
- **Key insight**: Local optimization leads to global catastrophe (starvation)

---

## 2. **Cognitive Heuristics**

### [A* Dispatch (Cost-Minimization)](ASTAR.md)
- **Type**: Urgency-weighted assignment
- **Complexity**: O(n²) per tick
- **Status**: ⚠️ Functional but suboptimal
- **Best for**: Light-load scenarios, scattered demand
- **Mean wait**: ~65–70 seconds ❌
- **Key insight**: Point-to-point assignment sacrifices throughput for responsiveness

### [A* + SCAN Hybrid](ASTAR_SCAN.md)
- **Type**: Hybrid cognitive + mechanical
- **Complexity**: O(n²) per tick
- **Status**: ✅ Recommended for most buildings
- **Best for**: Mixed traffic patterns, 24-hour buildings
- **Mean wait**: ~4–8 seconds ✅
- **Key insight**: Combines A*'s urgency-awareness with SCAN's sweep efficiency

---

## 3. **Reinforcement Learning**

### [Pure RL (CMA-ES / PPO)](CMAES.md)
- **Type**: Neural network policy learning
- **Complexity**: O(1) per tick (inference only)
- **Status**: ⚠️ Requires training
- **Best for**: Research, discovering novel strategies
- **Training cost**: 20–50 generations (~4–10 full runs)
- **Mean wait (untrained)**: 12,007 seconds ❌
- **Mean wait (trained)**: ~2–4 seconds ✅✅
- **Key insight**: Black-box learning exceeds heuristics, but interpretability and safety are concerns

---

## 4. **Hierarchical Neuro-Symbolic**

### [Meta-Controller (Final Solution)](META_CONTROLLER.md)
- **Type**: Neural supervisor + deterministic executors
- **Complexity**: O(1) meta-decision + O(n) execution
- **Status**: 🔬 Research prototype
- **Best for**: Production deployment with fairness + throughput requirements
- **Training cost**: 10–20 generations (~2–4 full runs)
- **Mean wait**: ~3–5 seconds ✅✅
- **Key insight**: Learns which proven algorithm to use at each moment (Pareto optimality)

---

## Performance Comparison Matrix

| Metric | SCAN | A*+Scan | Pure RL | Meta-Controller |
|--------|------|---------|---------|-----------------|
| **Mean wait** | 4–8s | 4–8s | 2–4s | **3–5s** |
| **Peak wait** | 120–180s | 90–120s | 40–70s | **40–60s** |
| **Training** | None | None | 50+ gens | **20 gens** |
| **Interpretable** | Yes | Yes | No | **Yes** |
| **Safe** | Yes | Yes | No | **Yes** |
| **Fairness** | High | High | Variable | **High** |
| **Throughput** | High | High | High | **High** |
| **Production-ready** | Yes | Yes | No | **Yes** |

---

## Traffic Regimen Performance

### Quiet Hours (Midnight–8 AM)
- **Winner**: A*+Scan or Meta-Controller
- **Why**: Fast response to individual requests matters more than throughput
- **Fairness index**: 95%+

### Morning Rush (8–10 AM)
- **Winner**: SCAN or Meta-Controller
- **Why**: Mechanical throughput is the only metric that prevents queue collapse
- **Mean wait**: 5–10 seconds
- **Fairness index**: 90%

### Lunch Chaos (12–2 PM)
- **Winner**: A*+Scan or Meta-Controller
- **Why**: Bidirectional scattered demand requires cognitive routing
- **Mean wait**: 8–12 seconds

### Evening Rush (5–7 PM)
- **Winner**: SCAN or Meta-Controller
- **Why**: Extreme downward demand requires maximum throughput
- **Mean wait**: 5–8 seconds
- **Fairness index**: 85%

---

## Algorithm Selection Guide

### Quick Decision Tree

```
Is the building experiencing rush hour?
├─ YES: Peak directional demand (morning/evening)?
│   ├─ YES: Use SCAN or Meta-Controller
│   └─ NO: Use A*+Scan or Meta-Controller
├─ NO: Light load? Scattered demand?
│   ├─ YES: Use A*+Scan
│   └─ NO: Any algorithm works well
```

### Deployment Recommendations

| Building Type | Recommended Algorithm | Rationale |
|---------------|----------------------|-----------|
| **Office tower** | A*+Scan | Mixed rush hours + lunch chaos |
| **Mall** | A*+Scan | Continuous scattered demand |
| **Hospital** | Meta-Controller | Safety-critical; needs adaptability |
| **Research facility** | Pure RL | Freedom to experiment; no strict SLA |
| **High-traffic transit** | SCAN | Throughput-first, predictable patterns |
| **Luxury residential** | Meta-Controller | Premium fairness + efficiency |

---

## Historical Context

### Phase 1: Baseline Algorithm (SCAN)
- Industry standard, proven reliable
- Poor during mixed-demand scenarios
- Mean wait: 40–60 seconds

### Phase 2: Greedy Improvement Attempt (Nearest First)
- ❌ Catastrophic failure
- Taught us: Local optimization ≠ global optimization

### Phase 3: Cognitive Targeting (A*)
- ✅ Urgency-aware assignment
- ⚠️ Poor throughput during rush hours
- Mean wait: 65–70 seconds

### Phase 4: Hybrid Approach (A*+Scan)
- ✅ Best practical algorithm for most buildings
- Combines SCAN's efficiency with A*'s fairness
- Mean wait: 4–8 seconds ✅

### Phase 5: Pure Learning (CMA-ES / PPO)
- ✅ Exceeds all heuristics when trained
- ❌ Black-box problem, regulatory risk
- Mean wait (trained): 2–4 seconds

### Phase 6: Neuro-Symbolic Hybrid (Meta-Controller)
- ✅✅ Best-in-class performance
- ✅ Interpretable and safe
- ✅ Converges quickly (only 20 generations)
- Mean wait: 3–5 seconds ✅✅
- **Recommended for production**

---

## The Fairness Paradox (Solved)

All algorithms face a fundamental trade-off:
- **Fair**: Rescue distant outliers → low throughput → queues explode
- **Efficient**: Maximize throughput → distant outliers starve → wait times spike

The **Meta-Controller solves this** by:
1. Detecting when the building is saturated (via implicit pattern learning)
2. Switching to SCAN mode (throughput-first) during saturation
3. Switching to A* mode (fairness-first) during quiet hours
4. Achieving **both** fairness AND efficiency across the entire day

Result: ~3–5 second mean wait with minimal starvation (peak wait < 60s).

---

## Implementation Notes

### Language: Python
- All controllers inherit from `BaseController` (agents.py)
- State vectors are 144-dimensional (full building information)
- Actions: MOVE_UP (0), MOVE_DOWN (1), STOP_OPEN (2), IDLE (3)

### Configuration (config.py)
```python
ALGORITHM = "astar_scan"  # Default recommended
TRAIN_RL = False          # Set True to enable RL training
SPEED = 1000              # Ticks per real second
```

### Model Files (models/)
- `cmaes_bank_a.npy` / `cmaes_bank_b.npy` — CMA-ES weights
- `ppo_bank_a.pt` / `ppo_bank_b.pt` — PPO network weights
- `meta_cmaes_bank_a_*` — Meta-controller weights

---

## References & Further Reading

### Academic Papers
- **SCAN/Elevator Algorithm**: Knobel, C., & Leurent, G. (1999). "Elevator Group Control System" IEEE Transactions on Control Systems Technology
- **CMA-ES**: Hansen, N., & Ostermeier, A. (2001). "Completely Derandomized Self-Adaptation in Evolution Strategies"
- **PPO**: Schulman, J., et al. (2017). "Proximal Policy Optimization Algorithms"
- **Hierarchical RL**: Kulkarni, T. D., et al. (2016). "Hierarchical Deep Reinforcement Learning"

### Simulation Papers
- **Discrete event simulation**: Law, A. M., & Kelton, W. D. (2015). "Simulation Modeling and Analysis"
- **Queuing theory**: Kleinrock, L. (1975). "Queuing Systems"

---

## Troubleshooting

### My algorithm is performing poorly
1. Check if it's trained (for RL algorithms, look for .npy files)
2. Verify state dimensions match (should be 144)
3. Ensure reward function weights are tuned (see CMAES.md)
4. Test on a shorter simulation first (easier to debug)

### Peak wait times are very high
- Try A*+Scan (best for mixed patterns)
- If already using A*+Scan, enable Meta-Controller
- Check if building state is saturating (>400 concurrent waiting)

### Algorithm keeps switching in logs (oscillation)
- For Meta-Controller: mode_signal is hovering near 0.0
- Retrain with higher sigma variance
- Consider soft-switching (weighted combination of policies)

---

## Contact & Contributions

For questions about specific algorithms or to propose improvements:
- See PROBLEM.md for system architecture overview
- Check individual algorithm files for implementation details
- Review simulation.py for integration points

## Last updated
2026-05-21
