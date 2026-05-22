# Nearest First (Greedy Search)

**Type**: Greedy Heuristic | **Implementation**: `NearestFirstController` | **Status**: Broken

## Overview
Nearest First is a pure greedy algorithm that assigns each idle elevator to the physically closest waiting passenger. While intuitive, it is rarely used in real elevators because it causes catastrophic starvation of distant floors.

## How It Works in Real Life

In theory, "always go to the closest passenger" seems efficient. In practice:
- An elevator would get trapped servicing repeatedly-pressing passengers on nearby floors
- Passengers on distant floors would wait indefinitely
- Multiple elevators would race to the same close floor, leaving other areas empty
- The system would thrash continuously, never reaching steady-state

This is why **no commercial elevator system uses pure Nearest First**.

## Implementation in Code (NearestFirstController)

### Control Logic
```python
def get_action_for_idle_lifts():
    for each idle elevator:
        # 1. Scan all unassigned hallway calls
        # 2. Calculate distance to each call
        distance = abs(lift.floor - call.floor)
        
        # 3. Assign this lift to the call with minimum distance
        closest_call = argmin(distance)
        
        # 4. That lift becomes "assigned" and travels to that floor
```

### Deconfliction Attempt
In your implementation, there is a deconfliction mechanism that counts how many lifts are already heading to each floor and only sends additional lifts if demand exceeds one lift's capacity (15 people). However, this is insufficient to prevent starvation during multi-directional traffic.

### Core Issue: Pure Distance-Only Greedy
- **No lookahead**: Doesn't batch multiple stops
- **No urgency**: A 10-minute-old call is treated the same as a fresh call
- **No momentum**: Recomputes assignment every tick, causing lifts to abandon good routes mid-sweep

## Mathematical Parameters
- **Single metric**: `cost = abs(lift.floor - call.floor)`
- Deconfliction: `demand_count_at_floor` (to avoid over-concentration)
- No weighting, no urgency factor, no direction awareness

## Accuracy to Academic Definition
True to Name: Yes. It is a pure **Greedy Search** algorithm. Every decision is locally optimal but globally suboptimal.

## Strengths
- Simple to implement
- Deterministic
- Low computational overhead
- Easy to debug

## Weaknesses
- Catastrophic starvation: Distant floors wait indefinitely
- Thrashing: Multiple lifts race to the same close floor
- No momentum: Lifts abandon current routes mid-sweep
- Unfair: Repeatedly pressing buttons traps the nearest elevator
- No direction awareness: Causes stutter loops and boarding rejections

## Performance Characteristics
- **Mean Wait Time**: ~34 seconds (deceptively good average)
- **Peak Wait Time**: 320+ seconds (catastrophic outliers)
- **Peak Concurrent Waiting**: 200+ passengers (massive queues form)
- **Energy Efficiency**: Poor (thrashing waste)
- **Fairness Index**: Extremely low (distant floors starved)

## Results (600-min run, 9:00 AM start, realistic traffic)

| Metric | Value |
|--------|-------|
| Delivered | ~8,100 |
| Mean avg wait | ~34 s |
| Peak max wait | ~320+ s |
| Peak waiting | ~200+ |
| Energy | ~295,000 |

## Why It Fails So Catastrophically

1. **Starvation Loop**: A person on Floor 2 repeatedly presses the button. The nearest idle elevator always goes to Floor 2. Meanwhile, a person on Floor 20 is trapped in an exponential wait.
2. **Thrashing**: Multiple idle elevators observe the same crowd on a nearby floor. All race to the same floor simultaneously, leaving distant areas completely empty.
3. **No Directional Memory**: Without momentum, lifts revert to pure distance calculations every tick, destroying sweep efficiency.
4. **Deconfliction Insufficient**: Even with demand-based deconfliction, a highly concentrated demand at a nearby floor will still starve distant floors.

## When Nearest First Might Work
- Only in scenarios with uniform, predictable demand across all floors
- Never in realistic multi-floor buildings with directional rush hours

## Conclusion

Nearest First is included as a **negative example** to demonstrate how naive greedy assignment fails catastrophically on multi-agent elevator problems, even without the extra complexity of passenger direction, door mechanics, or realistic traffic. It is a textbook case of a locally optimal strategy being globally terrible.

## Historical Note
This algorithm performs so poorly that it is deprecated in production code. It serves only as a baseline for teaching what **not** to do in elevator dispatching.

## Last updated
2026-05-21
