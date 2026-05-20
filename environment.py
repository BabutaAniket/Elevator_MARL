import random
import math
from enum import IntEnum
from dataclasses import dataclass, field
from typing import List, Dict, Set, Optional, Tuple
from collections import deque
import numpy as np

from config import (
    NUM_FLOORS, FLOOR_NAMES, FLOOR_INDEX,
    BANK_A_FLOOR_INDICES, BANK_B_FLOOR_INDICES,
    NUM_LIFTS_PER_BANK, MAX_CAPACITY, SAFE_CAPACITY,
    DOOR_OPEN_AVG, DOOR_OPEN_MAX,
    TRAVEL_TIME_PER_FLOOR, DECELERATION_TIME,
    STATE_DIM,
    MEAN_WAIT_BOUND, MAX_WAIT_BOUND,
    PEAK_FLOOR_WAIT_BOUND, DIRECTION_REVERSAL_COOLDOWN,
    ENERGY_BUDGET_PER_HOUR, MOVEMENT_BUDGET_PER_HOUR,
    ROLLING_WINDOW_TICKS,
)


class Direction(IntEnum):
    DOWN = -1
    IDLE = 0
    UP = 1


class LiftAction(IntEnum):
    MOVE_UP = 0
    MOVE_DOWN = 1
    STOP_OPEN = 2
    IDLE = 3


class LiftState(IntEnum):
    MOVING = 0
    DOOR_OPEN = 1
    IDLE = 2
    DECELERATING = 3


@dataclass
class Passenger:
    id: int
    origin: int
    destination: int
    arrival_time: int
    board_time: int = -1
    alight_time: int = -1
    assigned_bank: str = ''
    priority: bool = False

    @property
    def wait_time(self):
        if self.board_time < 0:
            return -1
        return self.board_time - self.arrival_time

    @property
    def journey_time(self):
        if self.alight_time < 0:
            return -1
        return self.alight_time - self.arrival_time

    @property
    def desired_direction(self) -> Direction:
        if self.destination > self.origin:
            return Direction.UP
        return Direction.DOWN


@dataclass
class Lift:
    name: str
    bank: str
    floor: int = 8
    direction: Direction = Direction.IDLE
    state: LiftState = LiftState.IDLE
    passengers: List[Passenger] = field(default_factory=list)
    door_timer: int = 0
    decel_timer: int = 0
    destinations: Set[int] = field(default_factory=set)
    floors_traveled: int = 0
    stops_made: int = 0
    idle_time: int = 0
    last_reversal_tick: int = -100
    hour_floors: int = 0
    hour_energy: float = 0.0
    hour_start_tick: int = 0

    @property
    def load(self):
        return len(self.passengers)

    @property
    def is_full(self):
        return self.load >= SAFE_CAPACITY

    @property
    def accessible_floors(self):
        if self.bank == 'A':
            return BANK_A_FLOOR_INDICES
        return BANK_B_FLOOR_INDICES


class LiftBank:

    def __init__(self, bank_id: str, lift_names: List[str]):
        self.bank_id = bank_id
        self.lifts = [Lift(name=n, bank=bank_id) for n in lift_names]
        self.accessible_floors = BANK_A_FLOOR_INDICES if bank_id == 'A' else BANK_B_FLOOR_INDICES


        self.hall_calls: Dict[int, List[Passenger]] = {f: [] for f in range(NUM_FLOORS)}
        self.recent_waits: deque = deque(maxlen=ROLLING_WINDOW_TICKS)

    def get_waiting_count(self, floor: int) -> int:
        return len(self.hall_calls.get(floor, []))

    def get_total_waiting(self) -> int:
        return sum(len(p) for p in self.hall_calls.values())

    def get_max_wait(self, floor: int, current_tick: int) -> float:
        plist = self.hall_calls.get(floor, [])
        if not plist:
            return 0.0
        return max(current_tick - p.arrival_time for p in plist)

    def get_rolling_avg_wait(self) -> float:
        if not self.recent_waits:
            return 0.0
        return sum(self.recent_waits) / len(self.recent_waits)

    def get_urgent_passengers(self, tick: int, threshold: int = 90) -> List[Tuple[int, Passenger]]:
        urgent = []
        for f, plist in self.hall_calls.items():
            for p in plist:
                if tick - p.arrival_time >= threshold:
                    urgent.append((f, p))
        urgent.sort(key=lambda x: -(tick - x[1].arrival_time))
        return urgent

    def get_floor_with_most_waiting(self) -> Tuple[int, int]:
        best_f, best_c = -1, 0
        for f, plist in self.hall_calls.items():
            if len(plist) > best_c:
                best_c = len(plist)
                best_f = f
        return best_f, best_c


class BuildingEnvironment:

    def __init__(self):
        from config import BANK_A_LIFT_NAMES, BANK_B_LIFT_NAMES
        self.bank_a = LiftBank('A', BANK_A_LIFT_NAMES)
        self.bank_b = LiftBank('B', BANK_B_LIFT_NAMES)
        self.tick = 0
        self.passenger_counter = 0
        self.delivered_passengers: List[Passenger] = []
        self.active_passengers: List[Passenger] = []
        self.total_energy = 0.0
        self.emergency_mode = False


        self.tick_pickups = 0
        self.tick_deliveries = 0
        self.tick_pickups_a = 0
        self.tick_deliveries_a = 0
        self.tick_pickups_b = 0
        self.tick_deliveries_b = 0
        self.tick_energy = 0.0

        # Set False for pure baseline algorithms (SCAN, Nearest, Round-Robin,
        # IdleWait) so the urgency dispatcher does not hijack their lifts.
        self.urgency_dispatch_enabled = True

    def reset(self):
        self.__init__()
        return self.get_state('A'), self.get_state('B')

    def add_passenger(self, origin: int, destination: int, priority: bool = False) -> Optional[Passenger]:
        if origin == destination:
            return None

        p = Passenger(
            id=self.passenger_counter,
            origin=origin,
            destination=destination,
            arrival_time=self.tick,
            priority=priority,
        )
        self.passenger_counter += 1


        bank = self._assign_bank(origin, destination)
        if bank is None:
            return None

        # Prevent "Passenger Death Spiral" by capping waiting passengers per floor
        if len(bank.hall_calls[origin]) > 20:
            return None  # Floor is too crowded, passenger gives up and takes stairs

        p.assigned_bank = bank.bank_id
        bank.hall_calls[origin].append(p)
        self.active_passengers.append(p)
        return p

    def _assign_bank(self, origin: int, destination: int) -> Optional[LiftBank]:
        a_can = (origin in BANK_A_FLOOR_INDICES and destination in BANK_A_FLOOR_INDICES)
        b_can = (origin in BANK_B_FLOOR_INDICES and destination in BANK_B_FLOOR_INDICES)

        if a_can and b_can:

            if self.bank_a.get_total_waiting() <= self.bank_b.get_total_waiting():
                return self.bank_a
            return self.bank_b
        elif a_can:
            return self.bank_a
        elif b_can:
            return self.bank_b
        return None

    def step(self, actions_a: List[int], actions_b: List[int]) -> Tuple[float, float, List[int], List[int]]:
        self.tick_pickups = 0
        self.tick_deliveries = 0
        self.tick_pickups_a = 0
        self.tick_deliveries_a = 0
        self.tick_pickups_b = 0
        self.tick_deliveries_b = 0
        self.tick_energy = 0.0
        self.tick_lift_deliveries_a = [0] * NUM_LIFTS_PER_BANK
        self.tick_lift_deliveries_b = [0] * NUM_LIFTS_PER_BANK
        self.tick_lift_pickups_a = [0] * NUM_LIFTS_PER_BANK
        self.tick_lift_pickups_b = [0] * NUM_LIFTS_PER_BANK

        safe_a = self._apply_safety_filter(self.bank_a, actions_a)
        safe_b = self._apply_safety_filter(self.bank_b, actions_b)

        self._execute_actions(self.bank_a, safe_a)
        self._execute_actions(self.bank_b, safe_b)

        reward_a = self._compute_reward(self.bank_a)
        reward_b = self._compute_reward(self.bank_b)

        self.tick += 1
        self.total_energy += self.tick_energy

        for bank in [self.bank_a, self.bank_b]:
            for lift in bank.lifts:
                if self.tick - lift.hour_start_tick >= 3600:
                    lift.hour_floors = 0
                    lift.hour_energy = 0.0
                    lift.hour_start_tick = self.tick

        return reward_a, reward_b, safe_a, safe_b

    def get_per_lift_rewards(self, bank: LiftBank) -> List[float]:
        """Per-lift reward for credit assignment. Only rewards the lift that did the work."""
        bid = bank.bank_id
        d_list = self.tick_lift_deliveries_a if bid == 'A' else self.tick_lift_deliveries_b
        p_list = self.tick_lift_pickups_a if bid == 'A' else self.tick_lift_pickups_b
        rewards = []
        for i, lift in enumerate(bank.lifts):
            energy = -0.01 * (1 + lift.load * 0.1) if lift.state == LiftState.MOVING else 0.0
            # Dense proximity shaping: reward being close to the most-urgent caller.
            # Gives non-zero gradient on navigation decisions, not just on rare deliveries.
            shaping = 0.0
            if not lift.passengers:  # empty lift — learn where to go
                best_wait = -1
                best_dist = 1
                for f, callers in bank.hall_calls.items():
                    if callers and f in lift.accessible_floors:
                        w = max(self.tick - p.arrival_time for p in callers)
                        if w > best_wait:
                            best_wait = w
                            best_dist = max(abs(lift.floor - f), 1)
                if best_wait >= 0:
                    shaping = 0.5 / best_dist  # 0.5 at same floor, tapers off with distance
            rewards.append(30.0 * d_list[i] + 10.0 * p_list[i] + energy + shaping)
        return rewards

    def _apply_safety_filter(self, bank: LiftBank, actions: List[int]) -> List[int]:
        safe = list(actions)
        accessible = bank.accessible_floors

        for i, lift in enumerate(bank.lifts):
            act = LiftAction(safe[i])

            if lift.state in (LiftState.DOOR_OPEN, LiftState.DECELERATING):
                safe[i] = int(LiftAction.IDLE)
                continue

            if act == LiftAction.MOVE_UP and lift.floor >= NUM_FLOORS - 1:
                safe[i] = int(LiftAction.IDLE)
            elif act == LiftAction.MOVE_DOWN and lift.floor <= 0:
                safe[i] = int(LiftAction.IDLE)

            if LiftAction(safe[i]) in (LiftAction.MOVE_UP, LiftAction.MOVE_DOWN):
                new_dir = Direction.UP if LiftAction(safe[i]) == LiftAction.MOVE_UP else Direction.DOWN
                
                # ─────────────────────────────────────────────────────────────────
                # THE KINEMATIC LOCK (Directional Collective Control)
                # Prevent any algorithm from doing a U-Turn mid-shaft if there 
                # is still work remaining in its current direction.
                # ─────────────────────────────────────────────────────────────────
                if lift.direction != Direction.IDLE and new_dir != lift.direction:
                    has_work_ahead = False
                    
                    if lift.direction == Direction.UP:
                        # Are there passengers inside going further up?
                        if any(d > lift.floor for d in lift.destinations):
                            has_work_ahead = True
                        # Are there people waiting in the hallway above us?
                        elif any(bank.get_waiting_count(f) > 0 for f in range(lift.floor + 1, NUM_FLOORS)):
                            has_work_ahead = True
                            
                    elif lift.direction == Direction.DOWN:
                        # Are there passengers inside going further down?
                        if any(d < lift.floor for d in lift.destinations):
                            has_work_ahead = True
                        # Are there people waiting in the hallway below us?
                        elif any(bank.get_waiting_count(f) > 0 for f in range(lift.floor - 1, -1, -1)):
                            has_work_ahead = True
                            
                    if has_work_ahead:
                        # Override the AI's attempt to reverse. Force it to idle this tick 
                        # so it can realize its mistake and continue its sweep next tick.
                        safe[i] = int(LiftAction.IDLE)

        # Opportunistic pickup: if an EMPTY lift is at a floor with waiting passengers,
        # stop immediately. Restricted to truly empty lifts to avoid disrupting lifts
        # that are already en route to their passengers' destinations.
        for i, lift in enumerate(bank.lifts):
            if lift.state in (LiftState.DOOR_OPEN, LiftState.DECELERATING):
                continue
            if lift.passengers or lift.destinations or lift.is_full:
                continue  # only apply to truly empty lifts
            if lift.floor not in accessible:
                continue
            pickable = any(p.destination in accessible
                           for p in bank.hall_calls.get(lift.floor, []))
            if pickable:
                safe[i] = int(LiftAction.STOP_OPEN)

        if self.urgency_dispatch_enabled:
            urgent_floors = set()
            for f, plist in bank.hall_calls.items():
                for p in plist:
                    w = self.tick - p.arrival_time
                    if w >= 20 or (p.priority and w >= 10):
                        urgent_floors.add(f)
                        break

            if urgent_floors:
                idle_lifts = [(i, bank.lifts[i]) for i in range(NUM_LIFTS_PER_BANK)
                              if bank.lifts[i].state == LiftState.IDLE
                              and bank.lifts[i].load == 0
                              and not bank.lifts[i].destinations
                              and LiftAction(safe[i]) == LiftAction.IDLE
                              # Skip lifts in reversal cooldown; they will naturally
                              # complete their direction change next tick without
                              # being hijacked by urgency dispatch.
                              and (self.tick - bank.lifts[i].last_reversal_tick)
                                  >= DIRECTION_REVERSAL_COOLDOWN]
                for uf in urgent_floors:
                    if not idle_lifts:
                        break
                    idle_lifts.sort(key=lambda x: abs(x[1].floor - uf))
                    idx, lift = idle_lifts.pop(0)
                    if uf > lift.floor:
                        safe[idx] = int(LiftAction.MOVE_UP)
                    elif uf < lift.floor:
                        safe[idx] = int(LiftAction.MOVE_DOWN)
                    else:
                        safe[idx] = int(LiftAction.STOP_OPEN)

        return safe

    def _execute_actions(self, bank: LiftBank, actions: List[int]):
        for i, lift in enumerate(bank.lifts):
            action = LiftAction(actions[i])
            self._execute_lift_action(bank, lift, action)

    def _execute_lift_action(self, bank: LiftBank, lift: Lift, action: LiftAction):

        if lift.state == LiftState.DOOR_OPEN:
            lift.door_timer -= 1
            if lift.door_timer <= 0:
                lift.state = LiftState.IDLE
            return


        if lift.state == LiftState.DECELERATING:
            lift.decel_timer -= 1
            if lift.decel_timer <= 0:
                lift.state = LiftState.IDLE
                self._open_doors(bank, lift)
            return

        if action == LiftAction.MOVE_UP:
            if lift.floor < NUM_FLOORS - 1:
                old_dir = lift.direction
                lift.floor += 1
                lift.direction = Direction.UP
                lift.state = LiftState.MOVING
                lift.floors_traveled += 1
                lift.hour_floors += 1
                e = 1.0 + 0.1 * lift.load
                self.tick_energy += e
                lift.hour_energy += e
                if old_dir == Direction.DOWN:
                    lift.last_reversal_tick = self.tick

        elif action == LiftAction.MOVE_DOWN:
            if lift.floor > 0:
                old_dir = lift.direction
                lift.floor -= 1
                lift.direction = Direction.DOWN
                lift.state = LiftState.MOVING
                lift.floors_traveled += 1
                lift.hour_floors += 1
                e = 1.0 + 0.1 * lift.load
                self.tick_energy += e
                lift.hour_energy += e
                if old_dir == Direction.UP:
                    lift.last_reversal_tick = self.tick

        elif action == LiftAction.STOP_OPEN:
            if lift.state == LiftState.MOVING:
                lift.state = LiftState.DECELERATING
                lift.decel_timer = DECELERATION_TIME
                self.tick_energy += 2.0
                lift.hour_energy += 2.0
            else:
                self._open_doors(bank, lift)

        elif action == LiftAction.IDLE:
            lift.state = LiftState.IDLE
            lift.direction = Direction.IDLE
            lift.idle_time += 1

    def _open_doors(self, bank: LiftBank, lift: Lift):
        lift.state = LiftState.DOOR_OPEN
        lift.stops_made += 1
        self.tick_energy += 2.0
        lift.hour_energy += 2.0

        lift_idx = bank.lifts.index(lift)

        alighting = [p for p in lift.passengers if p.destination == lift.floor]
        for p in alighting:
            p.alight_time = self.tick
            lift.passengers.remove(p)
            lift.destinations.discard(p.destination)
            self.delivered_passengers.append(p)
            self.active_passengers.remove(p)
            self.tick_deliveries += 1
            wait = p.board_time - p.arrival_time
            bank.recent_waits.append(wait)
            if bank.bank_id == 'A':
                self.tick_deliveries_a += 1
                self.tick_lift_deliveries_a[lift_idx] += 1
            else:
                self.tick_deliveries_b += 1
                self.tick_lift_deliveries_b[lift_idx] += 1

        waiting = bank.hall_calls[lift.floor][:]
        priority_first = sorted(waiting, key=lambda p: (
            not p.priority,
            -(self.tick - p.arrival_time),
        ))

        boarded = 0
        for p in priority_first:
            if lift.is_full:
                break
            if p.destination not in lift.accessible_floors:
                continue
            if lift.direction != Direction.IDLE and lift.load > 0:
                if p.desired_direction != lift.direction:
                    continue
            p.board_time = self.tick
            lift.passengers.append(p)
            lift.destinations.add(p.destination)
            bank.hall_calls[lift.floor].remove(p)
            self.tick_pickups += 1
            if bank.bank_id == 'A':
                self.tick_pickups_a += 1
                self.tick_lift_pickups_a[lift_idx] += 1
            else:
                self.tick_pickups_b += 1
                self.tick_lift_pickups_b[lift_idx] += 1
            boarded += 1
            # After the first passenger boards into an empty lift, immediately update
            # the lift's direction so subsequent passengers with the same desired
            # direction can also pass the directional boarding check above.
            if boarded == 1 and len(lift.passengers) == 1:
                dest = lift.passengers[0].destination
                if dest > lift.floor:
                    lift.direction = Direction.UP
                elif dest < lift.floor:
                    lift.direction = Direction.DOWN

        if lift.passengers and lift.direction == Direction.IDLE:
            avg_dest = sum(p.destination for p in lift.passengers) / len(lift.passengers)
            if avg_dest > lift.floor:
                lift.direction = Direction.UP
            elif avg_dest < lift.floor:
                lift.direction = Direction.DOWN

        exchanged = len(alighting) + boarded

        # If the lift stopped but exchanged 0 passengers due to a direction
        # mismatch (it has onboard passengers going one way, waiting passengers
        # going the other), force a quick 3-tick door close and reset direction
        # so the lift breaks out of the floor lock and keeps moving.
        if exchanged == 0 and lift.load > 0:
            lift.door_timer = 3
            lift.direction = Direction.IDLE
        else:
            lift.door_timer = min(max(3, 3 + exchanged), DOOR_OPEN_MAX)

    def _compute_reward(self, bank: LiftBank) -> float:
        """Unified reward with logarithmic base penalties and a starvation guardrail.

        Reward = W_DEL * Deliveries
               - W_ENG * Energy
               - 5 * ln(1 + wait_avg / 10)    [when passengers queue]
               - 5 * ln(1 + inside_avg / 10)  [when passengers ride]
               - starvation_penalty            [linear bleed if any wait > 300 s]

        The starvation guardrail overrides the flat log curve when a single
        passenger has been waiting more than 5 minutes, forcing the agent to
        service neglected floors rather than optimising the average.
        """
        W_DEL = 1.0
        W_ENG = 0.005  # slightly increased to balance the stronger wait penalties

        deliveries = self.tick_deliveries_a if bank.bank_id == 'A' else self.tick_deliveries_b

        # Energy consumed by this bank's lifts this tick
        tick_energy = 0.0
        for lift in bank.lifts:
            if lift.state == LiftState.MOVING:
                tick_energy += 1.0 + 0.1 * lift.load
            elif lift.state in (LiftState.DOOR_OPEN, LiftState.DECELERATING):
                tick_energy += 2.0

        # Calculate average AND max hall wait
        hall_waits = [
            self.tick - p.arrival_time
            for calls in bank.hall_calls.values()
            for p in calls
        ]
        wait_avg = float(np.mean(hall_waits)) if hall_waits else 0.0
        max_wait  = float(np.max(hall_waits))  if hall_waits else 0.0

        # Average time passengers are currently spending inside lifts
        inside_times = [
            self.tick - p.board_time
            for lift in bank.lifts
            for p in lift.passengers
            if p.board_time >= 0
        ]
        inside_avg = float(np.mean(inside_times)) if inside_times else 0.0

        # ─────────────────────────────────────────────────────────────────
        # STARVATION GUARDRAIL: linear penalty kicks in when any passenger
        # has waited more than 5 minutes (300 s), overriding the flat log
        # curve and forcing the agent to service neglected floors.
        # ─────────────────────────────────────────────────────────────────
        starvation_penalty = (max_wait - 300) / 50.0 if max_wait > 300 else 0.0

        reward = (W_DEL * deliveries
                  - W_ENG * tick_energy
                  - 5.0 * math.log1p(wait_avg / 10.0)
                  - 5.0 * math.log1p(inside_avg / 10.0)
                  - starvation_penalty)
        return reward

    def get_state(self, bank_id: str) -> np.ndarray:
        bank = self.bank_a if bank_id == 'A' else self.bank_b
        state = []

        for lift in bank.lifts:
            state.append(lift.floor / NUM_FLOORS)
            state.append(float(lift.direction))
            state.append(lift.load / SAFE_CAPACITY)
            state.append(1.0 if lift.state == LiftState.IDLE and lift.load == 0 else 0.0)

            dest_vec = [0.0] * NUM_FLOORS
            for d in lift.destinations:
                dest_vec[d] = 1.0
            state.extend(dest_vec)

        for f in range(NUM_FLOORS):
            up_count = sum(1 for p in bank.hall_calls[f] if p.destination > p.origin)
            down_count = sum(1 for p in bank.hall_calls[f] if p.destination < p.origin)
            max_wait = bank.get_max_wait(f, self.tick)
            state.append(math.log1p(up_count) / 5.0)
            state.append(math.log1p(down_count) / 5.0)
            state.append(min(max_wait / 300.0, 1.0))

        for f in range(NUM_FLOORS):
            state.append(min(bank.get_max_wait(f, self.tick) / 300.0, 1.0))

        sim_time = self.tick % 86400
        hour = sim_time / 3600.0
        state.append(math.sin(2 * math.pi * hour / 24.0))
        state.append(math.cos(2 * math.pi * hour / 24.0))
        state.append(min(bank.get_rolling_avg_wait() / 60.0, 1.0))
        state.append(min(bank.get_total_waiting() / 200.0, 1.0))

        arr = np.array(state, dtype=np.float32)
        expected = STATE_DIM
        if len(arr) != expected:
            if len(arr) < expected:
                arr = np.pad(arr, (0, expected - len(arr)))
            else:
                arr = arr[:expected]
        return arr

    def get_info(self) -> dict:
        return self._collect_info()

    def _collect_info(self) -> dict:
        all_waiting = []
        for bank in [self.bank_a, self.bank_b]:
            for calls in bank.hall_calls.values():
                for p in calls:
                    all_waiting.append(self.tick - p.arrival_time)

        recent_delivered = [p for p in self.delivered_passengers if p.alight_time >= self.tick - 60]

        # Avg journey time (board → alight) from recently delivered passengers (300-tick window)
        recent_window = [p for p in self.delivered_passengers if p.alight_time >= self.tick - 300]
        recent_journeys = [
            p.alight_time - p.board_time
            for p in recent_window
            if p.board_time >= 0 and p.alight_time > p.board_time
        ]
        avg_journey_time = float(np.mean(recent_journeys)) if recent_journeys else 0.0

        return {
            'tick': self.tick,
            'total_waiting': len(all_waiting),
            'avg_wait': np.mean(all_waiting) if all_waiting else 0,
            'max_wait': max(all_waiting) if all_waiting else 0,
            'p95_wait': np.percentile(all_waiting, 95) if all_waiting else 0,
            'total_delivered': len(self.delivered_passengers),
            'throughput_per_min': len(recent_delivered),
            'total_energy': self.total_energy,
            'tick_energy': self.tick_energy,
            'avg_journey_time': avg_journey_time,
            'pickups': self.tick_pickups,
            'deliveries': self.tick_deliveries,
            'bank_a_lifts': [self._lift_info(l) for l in self.bank_a.lifts],
            'bank_b_lifts': [self._lift_info(l) for l in self.bank_b.lifts],
            'floor_waiting': {
                FLOOR_NAMES[f]: self.bank_a.get_waiting_count(f) + self.bank_b.get_waiting_count(f)
                for f in range(NUM_FLOORS)
            }
        }

    def _lift_info(self, lift: Lift) -> dict:
        return {
            'name': lift.name,
            'bank': lift.bank,
            'floor': lift.floor,
            'floor_name': FLOOR_NAMES[lift.floor],
            'direction': int(lift.direction),
            'state': int(lift.state),
            'load': lift.load,
            'destinations': list(lift.destinations),
            'floors_traveled': lift.floors_traveled,
            'stops_made': lift.stops_made
        }

    def get_snapshot(self) -> dict:
        info = self._collect_info()
        info['active_passengers'] = len(self.active_passengers)
        return info
