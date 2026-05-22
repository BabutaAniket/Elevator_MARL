# A* Dispatch (Cost-Minimization Heuristic)

**Type**: Deterministic Heuristic | **Implementation**: `AStarDispatchController` | **Status**: Functional but suboptimal

## Overview

A* Dispatch is a cost-minimization algorithm inspired by modern "Destination Dispatch" systems in high-end office buildings. However, unlike true A* pathfinding, it performs cost-weighted assignment of idle elevators to waiting passengers using an urgency-weighted heuristic.

## How It Works in Real Life

Modern destination dispatch systems (common in corporate towers) use a central AI to analyze all waiting passengers and assign them to elevators based on:
- Physical distance to the passenger
- Passenger wait time (urgency)
- Elevator current load
- Building traffic patterns

The goal is to **minimize global average wait time** by optimally batching passengers into elevators, rather than having each elevator pursue an independent strategy.

## Implementation in Code (AStarDispatchController)

### Cost Function
```python
distance = abs(lift.floor - target_floor)
urgency = (current_tick - passenger.arrival_time) / 10.0
cost = distance - (urgency * 0.5)
```

### Assignment Algorithm
1. For every idle elevator, compute cost for every unassigned hallway call
2. Select the idle elevator + call pair with the **lowest cost**
3. Assign that lift to that call floor
4. Remove the call from unassigned list
5. Repeat until no more unassigned calls or idle lifts

### Key Insight: Urgency-Weighted Assignment
- A passenger who has waited 2 minutes gets a higher urgency score than a passenger who just arrived
- This forces the algorithm to "rescue" stranded passengers by mathematically reducing their cost
- However, unlike SCAN which rescues all passengers on a sweep, A* assigns lifts to individual floors

## Mathematical Parameters
```
distance_weight    = 1.0 (physical distance is fully weighted)
urgency_weight     = 0.5 (wait time reduces cost by 50% per 10-second interval)
load_penalty       = 0.3 (already-loaded lifts discouraged)
```

## Accuracy to Academic Definition
Not True to Name: In computer science, **A* (A-Star)** is a pathfinding algorithm used to navigate around obstacles in mazes. Elevators travel in straight 1D lines; there are no obstacles to path-find around. 

Your algorithm is actually a **Greedy Cost-Minimization Heuristic** or **Urgency-Weighted Cost Dispatcher**, not A*. While it uses heuristic principles similar to A*, it does not implement A* pathfinding. The name is a simplification for convenience.

## Strengths
- Urgency-aware: Prevents indefinite starvation
- Global optimization: Considers all lifts and all calls, not just per-lift greedy
- Deterministic: Reproducible results
- Moderate energy efficiency: Lifts idle when not needed

## Weaknesses
- Point-to-point dispatch: Assigns to individual floors, not sweeps
- Inefficient during multi-floor congestion: A lift serving one floor then repositioning is wasteful
- No directional momentum: Each lift operates independently on each tick
- Stutter loops possible: Without direction awareness, can trap elevators opening doors

## Performance Characteristics
- **Mean Wait Time**: ~65–70 seconds (significantly worse than SCAN)
- **Peak Wait Time**: ~180–220 seconds
- **Peak Concurrent Waiting**: ~400–450 passengers (massive queue buildup)
- **Energy Efficiency**: Good (lower throughput means idle periods)
- **Fairness Index**: Moderate (urgency prevents complete starvation, but assignment is greedy)

## Results (600-min run, 9:00 AM start, realistic traffic)

| Metric | Value |
|--------|-------|
| Delivered | ~6,800 |
| Mean avg wait | ~65–70 s |
| Peak waiting | ~400–450 |
| Energy | ~196,000 |

## Why It Underperforms vs SCAN

**The Fundamental Flaw: Point-to-Point Dispatch Instead of Sweeps**

During morning rush, parking floors P1–P7 all have 30–60 people each.
- **A* Dispatch**: Assigns one lift per floor → each lift serves 15 people per trip → 8 lifts × 15 = 120 people/cycle
- **SCAN**: Sends each lift on a full downward sweep, stopping at all 7 floors → ~60 people/lift/cycle

A* delivers roughly **4× less throughput** during peak periods.

### Example Scenario: Morning Rush
```
Waiting passengers:
  P7: 45 people
  P6: 50 people
  P5: 48 people
  P4: 40 people
  
SCAN behavior:
  - All 8 lifts start sweeping DOWN
  - Each hits all 4 busy floors in sequence
  - ~480 people delivered in 40 ticks
  
A* Dispatch behavior:
  - Lifts 1-4 assigned to P7, P6, P5, P4 respectively
  - Lifts 5-8 assigned to same floors (cost recomputation next tick)
  - Thrashing: lifts assign, move 1 tick, then reassign next tick
  - Poor throughput
```

## When A* Dispatch Works Well
- Low-load scenarios (< 50% capacity utilization)
- Uniform demand across all floors
- Bidirectional traffic (mixed up/down demand)
- Not suitable for rush hours (worst-case performance)

## Recommendations
- **Use SCAN** for predictable directional traffic (morning/evening rush)
- **Use A* + SCAN Hybrid** for mixed traffic conditions
- **Use pure A* only** for very light loads where high responsiveness is valued over throughput

## Conclusion

A* Dispatch demonstrates the importance of architectural decisions in elevator systems. While urgency-weighted cost assignment is intellectually appealing and works well in low-load scenarios, it fundamentally sacrifices throughput for responsiveness. During physical saturation (rush hours), this trade-off becomes catastrophic.

The lesson: **for multi-floor congestion, sweeping algorithms (SCAN) are superior to point-to-point assignment algorithms (A*), because they amortize travel and stopping costs across multiple passengers.**

## Last updated
2026-05-21

## Last updated
2026-05-05
