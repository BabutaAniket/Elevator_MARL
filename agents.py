import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from typing import List, Tuple, Dict
from collections import deque, Counter
import os

from config import (
    STATE_DIM, ACTION_DIM, NUM_LIFTS_PER_BANK,
    GAMMA, GAE_LAMBDA, PPO_CLIP, LEARNING_RATE,
    PPO_EPOCHS, BATCH_SIZE, NUM_FLOORS,
    BANK_A_FLOOR_INDICES, BANK_B_FLOOR_INDICES,
    FLOOR_INDEX, MAX_WAIT_BOUND, SAFE_CAPACITY,
    PEAK_FLOOR_WAIT_BOUND,
)
from environment import Direction, LiftState, LiftBank, LiftAction


def _heuristic_action(bank, lift, accessible):
    if lift.state in (LiftState.DOOR_OPEN, LiftState.DECELERATING):
        return int(LiftAction.IDLE)
    if lift.floor in lift.destinations:
        return int(LiftAction.STOP_OPEN)
    # Direction-aware pickup: a loaded lift travelling in a set direction only
    # collects passengers going the same way, preventing the door-stutter loop
    # where 0 passengers exchange and the lift is pinned at the floor.
    pickable = [
        p for p in bank.hall_calls[lift.floor]
        if p.destination in accessible
        and (
            lift.load == 0
            or lift.direction == Direction.IDLE
            or p.desired_direction == lift.direction
        )
    ]
    if pickable and not lift.is_full:
        return int(LiftAction.STOP_OPEN)
    if lift.destinations:
        nearest = min(lift.destinations, key=lambda d: abs(d - lift.floor))
        if nearest > lift.floor:
            return int(LiftAction.MOVE_UP)
        elif nearest < lift.floor:
            return int(LiftAction.MOVE_DOWN)
        else:
            return int(LiftAction.STOP_OPEN)
    return None


def _smart_heuristic(bank, lift, accessible, tick):
    ha = _heuristic_action(bank, lift, accessible)
    if ha is not None:
        return ha

    best_floor = -1
    best_score = -1e9
    for f in range(NUM_FLOORS):
        if f not in accessible:
            continue
        waiting = [p for p in bank.hall_calls[f] if p.destination in accessible]
        if not waiting:
            continue
        max_wait = max(tick - p.arrival_time for p in waiting)
        count = len(waiting)
        dist = max(abs(f - lift.floor), 1)
        score = count * 2.0 + max_wait / 10.0 - dist * 1.5
        if max_wait > MAX_WAIT_BOUND * 0.7:
            score += 100.0
        if score > best_score:
            best_score = score
            best_floor = f

    if best_floor >= 0:
        if best_floor > lift.floor:
            return int(LiftAction.MOVE_UP)
        elif best_floor < lift.floor:
            return int(LiftAction.MOVE_DOWN)
        else:
            return int(LiftAction.STOP_OPEN)
    return int(LiftAction.IDLE)


class ActorCritic(nn.Module):

    def __init__(self, state_dim, action_dim, num_lifts):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
        )
        self.actor_heads = nn.ModuleList([
            nn.Linear(128, action_dim) for _ in range(num_lifts)
        ])
        self.critic = nn.Linear(128, 1)

    def forward(self, state):
        features = self.shared(state)
        action_logits = [head(features) for head in self.actor_heads]
        value = self.critic(features)
        return action_logits, value

    def get_actions(self, state):
        with torch.no_grad():
            s = torch.FloatTensor(state).unsqueeze(0)
            logits_list, value = self(s)
            actions = []
            log_probs = []
            for logits in logits_list:
                dist = Categorical(logits=logits.squeeze(0))
                action = dist.sample()
                actions.append(action.item())
                log_probs.append(dist.log_prob(action).item())
            return actions, log_probs, value.item()

    def evaluate(self, states, actions_list):
        logits_list, values = self(states)
        total_log_probs = torch.zeros(states.shape[0])
        total_entropy = torch.zeros(states.shape[0])
        for i, logits in enumerate(logits_list):
            dist = Categorical(logits=logits)
            a = actions_list[:, i]
            total_log_probs += dist.log_prob(a)
            total_entropy += dist.entropy()
        return total_log_probs, values.squeeze(-1), total_entropy / len(logits_list)


class PPOAgent:

    def __init__(self, bank_id):
        self.bank_id = bank_id
        self.model = ActorCritic(STATE_DIM, ACTION_DIM, NUM_LIFTS_PER_BANK)
        self.optimizer = optim.Adam(self.model.parameters(), lr=LEARNING_RATE)
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []
        self.reward_normalizer = RunningRewardNormalizer()

    def select_action(self, state, bank=None):
        with torch.no_grad():
            s = torch.FloatTensor(state).unsqueeze(0)
            logits_list, value = self.model(s)

        raw_actions = []
        for logits in logits_list:
            dist = Categorical(logits=logits.squeeze(0))
            action = dist.sample()
            raw_actions.append(action.item())

        if bank:
            actions = []
            for i, lift in enumerate(bank.lifts):
                if lift.state in (LiftState.DOOR_OPEN, LiftState.DECELERATING):
                    actions.append(int(LiftAction.IDLE))
                elif lift.floor in lift.destinations:
                    actions.append(int(LiftAction.STOP_OPEN))
                elif lift.passengers and lift.destinations:
                    nearest = min(lift.destinations, key=lambda d: abs(d - lift.floor))
                    if nearest > lift.floor:
                        actions.append(int(LiftAction.MOVE_UP))
                    else:
                        actions.append(int(LiftAction.MOVE_DOWN))
                else:
                    actions.append(raw_actions[i])
        else:
            actions = raw_actions

        # Compute log_probs for the EXECUTED actions (not raw NN outputs)
        log_probs = []
        for i, logits in enumerate(logits_list):
            dist = Categorical(logits=logits.squeeze(0))
            log_probs.append(dist.log_prob(torch.tensor(actions[i])).item())

        self.states.append(state)
        self.actions.append(actions)
        self.log_probs.append(log_probs)
        self.values.append(value.item())
        return actions

    def store_reward(self, reward, done=False):
        self.rewards.append(reward)
        self.dones.append(done)

    def update(self):
        if len(self.states) < 32:
            return 0.0

        states = torch.FloatTensor(np.array(self.states))
        actions = torch.LongTensor(np.array(self.actions))
        old_log_probs = torch.FloatTensor(np.array(self.log_probs)).sum(dim=-1)
        raw_rewards = np.array(self.rewards)
        rewards = np.array([self.reward_normalizer.normalize(r) for r in raw_rewards])
        values = np.array(self.values)
        dones = np.array(self.dones, dtype=np.float32)

        advantages = np.zeros_like(rewards)
        last_gae = 0
        for t in reversed(range(len(rewards))):
            next_val = values[t + 1] if t < len(rewards) - 1 else 0
            delta = rewards[t] + GAMMA * next_val * (1 - dones[t]) - values[t]
            last_gae = delta + GAMMA * GAE_LAMBDA * (1 - dones[t]) * last_gae
            advantages[t] = last_gae

        returns = advantages + values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        advantages = torch.FloatTensor(advantages)
        returns = torch.FloatTensor(returns)

        total_loss = 0
        for _ in range(PPO_EPOCHS):
            indices = np.random.permutation(len(states))
            for start in range(0, len(states), BATCH_SIZE):
                idx = indices[start:start + BATCH_SIZE]
                batch_states = states[idx]
                batch_actions = actions[idx]
                batch_old_lp = old_log_probs[idx]
                batch_adv = advantages[idx]
                batch_returns = returns[idx]

                new_lp, new_values, entropy = self.model.evaluate(batch_states, batch_actions)
                ratio = torch.exp(new_lp - batch_old_lp)
                surr1 = ratio * batch_adv
                surr2 = torch.clamp(ratio, 1 - PPO_CLIP, 1 + PPO_CLIP) * batch_adv
                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = 0.5 * (new_values - batch_returns).pow(2).mean()
                entropy_bonus = -0.05 * entropy.mean()

                loss = actor_loss + critic_loss + entropy_bonus
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                self.optimizer.step()
                total_loss += loss.item()

        self.states.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.rewards.clear()
        self.values.clear()
        self.dones.clear()
        return total_loss

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        if os.path.exists(path):
            try:
                self.model.load_state_dict(torch.load(path, weights_only=True))
            except (RuntimeError, KeyError):
                pass


class ScanController:

    def __init__(self, bank):
        self.bank = bank
        self.lift_dirs = [Direction.UP] * NUM_LIFTS_PER_BANK

    def get_actions(self, bank, tick):
        actions = []
        accessible = bank.accessible_floors
        for i, lift in enumerate(bank.lifts):
            if lift.state in (LiftState.DOOR_OPEN, LiftState.DECELERATING):
                actions.append(int(LiftAction.IDLE))
                continue

            has_dropoff = lift.floor in lift.destinations
            # Direction-aware pickup: only stop for passengers going the same way
            # as the lift's current sweep direction when the lift already has riders.
            lift_dir = self.lift_dirs[i]
            has_pickup = (
                lift.floor in accessible
                and not lift.is_full
                and any(
                    p.destination in accessible
                    and (lift.load == 0 or p.desired_direction == lift_dir)
                    for p in bank.hall_calls[lift.floor]
                )
            )

            if has_dropoff or has_pickup:
                actions.append(int(LiftAction.STOP_OPEN))
                continue

            d = self.lift_dirs[i]
            has_work_ahead = False
            if d == Direction.UP:
                for f in range(lift.floor + 1, NUM_FLOORS):
                    if f not in accessible:
                        continue
                    if f in lift.destinations or len([p for p in bank.hall_calls[f]
                                                      if p.destination in accessible]) > 0:
                        has_work_ahead = True
                        break
            else:
                for f in range(lift.floor - 1, -1, -1):
                    if f not in accessible:
                        continue
                    if f in lift.destinations or len([p for p in bank.hall_calls[f]
                                                      if p.destination in accessible]) > 0:
                        has_work_ahead = True
                        break

            if not has_work_ahead:
                if d == Direction.UP:
                    self.lift_dirs[i] = Direction.DOWN
                    for f in range(lift.floor - 1, -1, -1):
                        if f not in accessible:
                            continue
                        if f in lift.destinations or len([p for p in bank.hall_calls[f]
                                                          if p.destination in accessible]) > 0:
                            has_work_ahead = True
                            break
                else:
                    self.lift_dirs[i] = Direction.UP
                    for f in range(lift.floor + 1, NUM_FLOORS):
                        if f not in accessible:
                            continue
                        if f in lift.destinations or len([p for p in bank.hall_calls[f]
                                                          if p.destination in accessible]) > 0:
                            has_work_ahead = True
                            break

            if not has_work_ahead:
                actions.append(int(LiftAction.IDLE))
            elif self.lift_dirs[i] == Direction.UP:
                actions.append(int(LiftAction.MOVE_UP))
            else:
                actions.append(int(LiftAction.MOVE_DOWN))

        return actions


class NearestFirstController:

    def get_actions(self, bank, tick):
        actions = [int(LiftAction.IDLE)] * NUM_LIFTS_PER_BANK
        accessible = bank.accessible_floors
        floor_assignments = {}

        for i, lift in enumerate(bank.lifts):
            if lift.state in (LiftState.DOOR_OPEN, LiftState.DECELERATING):
                continue

            if lift.floor in lift.destinations:
                actions[i] = int(LiftAction.STOP_OPEN)
                continue

            # Direction-aware pickup: loaded lifts only collect same-direction riders.
            has_pickup = (
                not lift.is_full
                and any(
                    p.destination in accessible
                    and (
                        lift.load == 0
                        or lift.direction == Direction.IDLE
                        or p.desired_direction == lift.direction
                    )
                    for p in bank.hall_calls[lift.floor]
                )
            )
            if has_pickup:
                actions[i] = int(LiftAction.STOP_OPEN)
                continue

            if lift.destinations:
                min_dest = min(lift.destinations)
                max_dest = max(lift.destinations)
                if lift.floor <= min_dest:
                    actions[i] = int(LiftAction.MOVE_UP)
                elif lift.floor >= max_dest:
                    actions[i] = int(LiftAction.MOVE_DOWN)
                else:
                    nearest = min(lift.destinations, key=lambda d: abs(d - lift.floor))
                    if nearest > lift.floor:
                        actions[i] = int(LiftAction.MOVE_UP)
                    else:
                        actions[i] = int(LiftAction.MOVE_DOWN)
                continue

            nearest_floor = -1
            nearest_dist = NUM_FLOORS + 1
            for f in range(NUM_FLOORS):
                if f not in accessible:
                    continue
                waiting = [p for p in bank.hall_calls[f] if p.destination in accessible]
                wcount = len(waiting)
                if wcount <= 0:
                    continue
                assigned = floor_assignments.get(f, 0)
                lifts_needed = max(1, (wcount + 14) // 15)
                if assigned >= lifts_needed:
                    continue
                d = abs(f - lift.floor)
                if d < nearest_dist:
                    nearest_dist = d
                    nearest_floor = f

            if nearest_floor >= 0:
                floor_assignments[nearest_floor] = floor_assignments.get(nearest_floor, 0) + 1
                if nearest_floor > lift.floor:
                    actions[i] = int(LiftAction.MOVE_UP)
                elif nearest_floor < lift.floor:
                    actions[i] = int(LiftAction.MOVE_DOWN)

        return actions


class AStarDispatchController:

    def get_actions(self, bank, tick):
        actions = [int(LiftAction.IDLE)] * NUM_LIFTS_PER_BANK
        accessible = bank.accessible_floors
        floor_assigned: dict = {}  # floor -> lifts already committed this tick

        # Pre-register floors that are actively being served by lifts currently
        # opening doors or decelerating.  Without this, those floors look
        # completely "unassigned" and every other lift dogpiles onto them.
        for lift in bank.lifts:
            if lift.state in (LiftState.DOOR_OPEN, LiftState.DECELERATING):
                floor_assigned[lift.floor] = floor_assigned.get(lift.floor, 0) + 1

        for i, lift in enumerate(bank.lifts):
            if lift.state in (LiftState.DOOR_OPEN, LiftState.DECELERATING):
                continue

            if lift.floor in lift.destinations:
                actions[i] = int(LiftAction.STOP_OPEN)
                continue

            pickable = [p for p in bank.hall_calls[lift.floor]
                        if p.destination in accessible]
            if pickable and not lift.is_full:
                actions[i] = int(LiftAction.STOP_OPEN)
                continue

            if lift.destinations:
                nearest = min(lift.destinations,
                              key=lambda d: abs(d - lift.floor))
                actions[i] = (int(LiftAction.MOVE_UP) if nearest > lift.floor
                              else int(LiftAction.MOVE_DOWN))
                # Also register the destination so other lifts don't pile on
                floor_assigned[nearest] = floor_assigned.get(nearest, 0) + 1
                continue

            best_floor = -1
            best_cost = float('inf')
            for f in range(NUM_FLOORS):
                if f not in accessible:
                    continue
                waiting = [p for p in bank.hall_calls[f]
                           if p.destination in accessible]
                if not waiting:
                    continue
                max_wait = max(tick - p.arrival_time for p in waiting)
                travel = abs(f - lift.floor)
                urgency = max_wait * 0.3  # no cap: extreme waits dominate proximity
                # Penalise floors already targeted by earlier lifts this tick
                # so lifts spread out rather than all converging on one floor
                oversubscription = floor_assigned.get(f, 0) * (urgency * 0.5 + 1)
                cost = travel - urgency + oversubscription
                if cost < best_cost:
                    best_cost = cost
                    best_floor = f

            if best_floor >= 0:
                floor_assigned[best_floor] = floor_assigned.get(best_floor, 0) + 1
                if best_floor > lift.floor:
                    actions[i] = int(LiftAction.MOVE_UP)
                elif best_floor < lift.floor:
                    actions[i] = int(LiftAction.MOVE_DOWN)

        return actions


class AStarScanController:

    def __init__(self):
        self._dir = {}

    def get_actions(self, bank, tick):
        actions = [int(LiftAction.IDLE)] * NUM_LIFTS_PER_BANK
        accessible = bank.accessible_floors

        for i, lift in enumerate(bank.lifts):
            if lift.state in (LiftState.DOOR_OPEN, LiftState.DECELERATING):
                continue

            if lift.floor in lift.destinations:
                actions[i] = int(LiftAction.STOP_OPEN)
                continue

            d = self._dir.get(i, 0)

            # FIX 1: Direction-Aware Pickup Check to prevent infinite door-stuttering loops
            has_matching_pickup = False
            if lift.floor in accessible and not lift.is_full:
                for p in bank.hall_calls[lift.floor]:
                    if p.destination in accessible:
                        # If empty or idle, any pickup is valid
                        if lift.load == 0 or d == 0:
                            has_matching_pickup = True
                            break
                        # If moving, only stop for passengers traveling in our direction
                        elif (d > 0 and p.destination > lift.floor) or (d < 0 and p.destination < lift.floor):
                            has_matching_pickup = True
                            break

            if has_matching_pickup:
                actions[i] = int(LiftAction.STOP_OPEN)
                continue

            if lift.destinations:
                nearest = min(lift.destinations, key=lambda f: abs(f - lift.floor))
                d = 1 if nearest > lift.floor else -1
                self._dir[i] = d
            else:
                if d != 0 and not self._has_work_in_dir(bank, lift, d, accessible):
                    d = 0

                if d == 0:
                    d = self._astar_direction(bank, lift, accessible, tick)
                    self._dir[i] = d

                if d == 0:
                    actions[i] = int(LiftAction.IDLE)
                    continue

                if not self._has_work_in_dir(bank, lift, d, accessible):
                    d = -d
                    self._dir[i] = d
                    if not self._has_work_in_dir(bank, lift, d, accessible):
                        actions[i] = int(LiftAction.IDLE)
                        continue

            actions[i] = int(LiftAction.MOVE_UP) if d > 0 else int(LiftAction.MOVE_DOWN)

        return actions

    def _has_work_in_dir(self, bank, lift, direction, accessible):
        rng = (range(lift.floor + 1, NUM_FLOORS) if direction > 0
               else range(lift.floor - 1, -1, -1))
        for f in rng:
            if f not in accessible:
                continue
            if f in lift.destinations:
                return True
            if any(p.destination in accessible for p in bank.hall_calls[f]):
                return True
        return False

    def _astar_direction(self, bank, lift, accessible, tick):
        up_score = 0.0
        dn_score = 0.0
        for f in range(NUM_FLOORS):
            if f not in accessible:
                continue
            waiting = [p for p in bank.hall_calls[f] if p.destination in accessible]
            if not waiting:
                continue

            max_wait = max(tick - p.arrival_time for p in waiting)

            # FIX 2: Linear cost scoring prevents mathematical distortion over long runs
            urgency = (max_wait / 10.0) + (len(waiting) * 2.0)
            dist = max(abs(f - lift.floor), 1)
            score = urgency - (dist * 1.5)

            if f > lift.floor:
                up_score += max(score, 0)
            else:
                dn_score += max(score, 0)

        if up_score == 0 and dn_score == 0:
            return 0
        return 1 if up_score > dn_score else -1



class DQNetwork(nn.Module):
    """Dueling DQN: splits Q(s,a) into V(s) + A(s,a) - mean(A)."""

    LIFT_FEAT_DIM = 8

    def __init__(self, state_dim, action_dim, num_lifts):
        super().__init__()
        self.num_lifts = num_lifts
        self.action_dim = action_dim

        self.global_encoder = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
        )

        per_lift_input = 128 + self.LIFT_FEAT_DIM

        # Dueling streams: separate value and advantage heads per lift
        self.value_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(per_lift_input, 64),
                nn.ReLU(),
                nn.Linear(64, 1),
            ) for _ in range(num_lifts)
        ])
        self.advantage_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(per_lift_input, 64),
                nn.ReLU(),
                nn.Linear(64, action_dim),
            ) for _ in range(num_lifts)
        ])

    def forward(self, state, lift_features=None):
        global_feat = self.global_encoder(state)
        results = []
        for i in range(self.num_lifts):
            if lift_features is not None:
                lf = lift_features[:, i, :]
            else:
                lf = torch.zeros(state.shape[0], self.LIFT_FEAT_DIM)
            combined = torch.cat([global_feat, lf], dim=-1)
            v = self.value_heads[i](combined)          # (batch, 1)
            a = self.advantage_heads[i](combined)      # (batch, action_dim)
            q = v + a - a.mean(dim=-1, keepdim=True)   # Dueling aggregation
            results.append(q)
        return results


class RunningRewardNormalizer:
    """Tracks running mean/var of rewards for adaptive scaling."""
    def __init__(self, clip=10.0):
        self.mean = 0.0
        self.var = 1.0
        self.count = 1e-4
        self.clip = clip

    def normalize(self, reward):
        self.count += 1
        delta = reward - self.mean
        self.mean += delta / self.count
        delta2 = reward - self.mean
        self.var += (delta * delta2 - self.var) / self.count
        std = max(self.var ** 0.5, 1e-6)
        return max(-self.clip, min(self.clip, (reward - self.mean) / std))


def _extract_lift_features(bank, tick):
    feats = []
    for lift in bank.lifts:
        f = [
            lift.floor / NUM_FLOORS,
            float(lift.direction),
            lift.load / SAFE_CAPACITY,
            1.0 if lift.state == LiftState.IDLE else 0.0,
            1.0 if lift.state == LiftState.DOOR_OPEN else 0.0,
            len(lift.destinations) / NUM_FLOORS,
            lift.hour_floors / 600.0 if hasattr(lift, 'hour_floors') else 0.0,
            min(lift.idle_time / 100.0, 1.0),
        ]
        feats.append(f)
    return np.array(feats, dtype=np.float32)


class DQNAgent:

    def __init__(self, bank_id):
        self.bank_id = bank_id
        self.q_net = DQNetwork(STATE_DIM, ACTION_DIM, NUM_LIFTS_PER_BANK)
        self.target_net = DQNetwork(STATE_DIM, ACTION_DIM, NUM_LIFTS_PER_BANK)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=LEARNING_RATE)

        self.replay_buffer = deque(maxlen=100000)
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.998
        self.target_update_freq = 250
        self.step_count = 0
        self.last_state = None
        self.last_actions = None
        self.last_lift_feats = None
        self.reward_normalizer = RunningRewardNormalizer()

    def select_action(self, state, bank=None):
        self.last_state = state
        accessible = bank.accessible_floors if bank else set()
        tick = 0

        if bank:
            self.last_lift_feats = _extract_lift_features(bank, tick)
        else:
            self.last_lift_feats = np.zeros((NUM_LIFTS_PER_BANK, DQNetwork.LIFT_FEAT_DIM), dtype=np.float32)

        if random.random() < self.epsilon:
            raw_actions = [random.randint(0, ACTION_DIM - 1) for _ in range(NUM_LIFTS_PER_BANK)]
        else:
            with torch.no_grad():
                s = torch.FloatTensor(state).unsqueeze(0)
                lf = torch.FloatTensor(self.last_lift_feats).unsqueeze(0)
                q_values = self.q_net(s, lf)
                raw_actions = [qv.argmax(dim=-1).item() for qv in q_values]

        if bank:
            actions = []
            for i, lift in enumerate(bank.lifts):
                if lift.state in (LiftState.DOOR_OPEN, LiftState.DECELERATING):
                    actions.append(int(LiftAction.IDLE))
                elif lift.floor in lift.destinations:
                    actions.append(int(LiftAction.STOP_OPEN))
                elif lift.passengers and lift.destinations:
                    nearest = min(lift.destinations, key=lambda d: abs(d - lift.floor))
                    if nearest > lift.floor:
                        actions.append(int(LiftAction.MOVE_UP))
                    else:
                        actions.append(int(LiftAction.MOVE_DOWN))
                else:
                    # Empty lift with no in-progress destination:
                    # Check if there are pickable passengers at the current floor
                    pickable = not lift.is_full and any(
                        p.destination in accessible
                        for p in bank.hall_calls.get(lift.floor, []))
                    if pickable:
                        actions.append(int(LiftAction.STOP_OPEN))
                    else:
                        act = raw_actions[i]
                        # Block stuck-at-empty-floor cycle:
                        # Q-value collapse makes DQN always pick STOP_OPEN, which
                        # opens empty doors forever and also blocks urgency dispatch
                        # (urgency dispatch only overrides IDLE, not STOP_OPEN)
                        if act == int(LiftAction.STOP_OPEN) and not lift.passengers:
                            act = int(LiftAction.IDLE)
                        actions.append(act)
        else:
            actions = raw_actions

        self.last_actions = actions
        return actions

    def store_reward(self, per_lift_rewards, next_state=None, bank=None, done=False):
        if self.last_state is not None and next_state is not None and bank is not None:
            # Extract the new lift features from the next state
            next_lift_feats = _extract_lift_features(bank, 0)
            # per_lift_rewards: list of 8 individual lift rewards (fixes multi-agent credit assignment)
            self.replay_buffer.append(
                (self.last_state, self.last_lift_feats, self.last_actions,
                 per_lift_rewards, next_state, next_lift_feats, done))

    def update(self):
        if len(self.replay_buffer) < BATCH_SIZE * 2:
            return 0.0

        batch = random.sample(list(self.replay_buffer), BATCH_SIZE)
        states = torch.FloatTensor(np.array([b[0] for b in batch]))
        lift_feats = torch.FloatTensor(np.array([b[1] for b in batch]))
        actions = [b[2] for b in batch]
        # per_lift_rewards_batch: (BATCH_SIZE, NUM_LIFTS_PER_BANK)
        per_lift_rewards_batch = np.array([b[3] for b in batch], dtype=np.float32)
        next_states = torch.FloatTensor(np.array([b[4] for b in batch]))
        next_lift_feats = torch.FloatTensor(np.array([b[5] for b in batch]))
        dones = torch.FloatTensor([b[6] for b in batch])

        q_values = self.q_net(states, lift_feats)

        with torch.no_grad():
            # Use next_lift_feats to ensure network sees the correct next state features
            next_q_online = self.q_net(next_states, next_lift_feats)
            next_q_target = self.target_net(next_states, next_lift_feats)

        total_loss = torch.tensor(0.0)
        for lift_i in range(NUM_LIFTS_PER_BANK):
            # Use THIS lift's individual reward — fixes multi-agent credit assignment
            raw_rewards_i = per_lift_rewards_batch[:, lift_i]
            rewards_i = torch.FloatTensor(
                [self.reward_normalizer.normalize(float(r)) for r in raw_rewards_i])

            a = torch.LongTensor([act[lift_i] for act in actions])
            current_q = q_values[lift_i].gather(1, a.unsqueeze(1)).squeeze(1)
            best_next_a = next_q_online[lift_i].argmax(dim=-1)
            max_next_q = next_q_target[lift_i].gather(1, best_next_a.unsqueeze(1)).squeeze(1)
            target_q = rewards_i + GAMMA * max_next_q * (1 - dones)
            total_loss = total_loss + nn.functional.smooth_l1_loss(current_q, target_q.detach())

        self.optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), 10.0)
        self.optimizer.step()

        self.step_count += 1
        if self.step_count % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        return total_loss.item()

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'model': self.q_net.state_dict(),
            'epsilon': self.epsilon,
            'step_count': self.step_count,
        }, path)

    def load(self, path):
        if os.path.exists(path):
            try:
                ckpt = torch.load(path, weights_only=False)
                if isinstance(ckpt, dict) and 'model' in ckpt:
                    self.q_net.load_state_dict(ckpt['model'])
                    self.epsilon = ckpt.get('epsilon', self.epsilon_min)
                    self.step_count = ckpt.get('step_count', 0)
                else:
                    self.q_net.load_state_dict(ckpt)
                self.target_net.load_state_dict(self.q_net.state_dict())
            except (RuntimeError, KeyError):
                pass


class PolicyNetwork(nn.Module):

    def __init__(self, state_dim, action_dim, num_lifts):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
        )
        self.heads = nn.ModuleList([
            nn.Linear(128, action_dim) for _ in range(num_lifts)
        ])

    def forward(self, state):
        features = self.shared(state)
        return [head(features) for head in self.heads]


class REINFORCEAgent:

    def __init__(self, bank_id):
        self.bank_id = bank_id
        self.policy = PolicyNetwork(STATE_DIM, ACTION_DIM, NUM_LIFTS_PER_BANK)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=LEARNING_RATE * 0.5)
        self.saved_log_probs = []
        self.rewards = []

    def select_action(self, state, bank=None):
        s = torch.FloatTensor(state).unsqueeze(0)
        logits_list = self.policy(s)
        raw_actions = []
        lp_sum = torch.tensor(0.0)
        for logits in logits_list:
            dist = Categorical(logits=logits.squeeze(0))
            action = dist.sample()
            lp_sum = lp_sum + dist.log_prob(action)
            raw_actions.append(action.item())
        self.saved_log_probs.append(lp_sum)

        if bank:
            actions = []
            for i, lift in enumerate(bank.lifts):
                if lift.state in (LiftState.DOOR_OPEN, LiftState.DECELERATING):
                    actions.append(int(LiftAction.IDLE))
                elif lift.floor in lift.destinations:
                    actions.append(int(LiftAction.STOP_OPEN))
                elif lift.passengers and lift.destinations:
                    nearest = min(lift.destinations, key=lambda d: abs(d - lift.floor))
                    if nearest > lift.floor:
                        actions.append(int(LiftAction.MOVE_UP))
                    else:
                        actions.append(int(LiftAction.MOVE_DOWN))
                else:
                    actions.append(raw_actions[i])
            return actions
        return raw_actions

    def store_reward(self, reward, done=False):
        self.rewards.append(reward)

    def update(self):
        if len(self.rewards) < 32:
            return 0.0

        R = 0
        returns = []
        for r in reversed(self.rewards):
            R = r + GAMMA * R
            returns.insert(0, R)

        returns = torch.FloatTensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        loss = torch.tensor(0.0)
        for lp, G in zip(self.saved_log_probs, returns):
            loss = loss - lp * G

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
        self.optimizer.step()

        self.saved_log_probs.clear()
        self.rewards.clear()
        return loss.item()

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.policy.state_dict(), path)

    def load(self, path):
        if os.path.exists(path):
            try:
                self.policy.load_state_dict(torch.load(path, weights_only=True))
            except (RuntimeError, KeyError):
                pass


class QLearningController:

    def __init__(self, bank_id):
        self.bank_id = bank_id
        self.q_table = {}
        self.alpha = 0.1
        self.epsilon = 0.3
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.9998
        self.last_state_key = None
        self.last_actions = None

    def _discretize(self, bank, tick):
        avg_floor = sum(l.floor for l in bank.lifts) / NUM_LIFTS_PER_BANK
        floor_bin = int(avg_floor / (NUM_FLOORS / 4))
        total_w = bank.get_total_waiting()
        wait_bin = min(total_w // 5, 10)
        idle_count = sum(1 for l in bank.lifts if l.state == LiftState.IDLE and l.load == 0)
        hour = (tick % 86400) / 3600
        time_bin = int(hour // 2)
        return (floor_bin, wait_bin, idle_count, time_bin)

    def select_action(self, bank, tick):
        state_key = self._discretize(bank, tick)
        self.last_state_key = state_key

        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(ACTION_DIM)

        q_vals = self.q_table[state_key]
        accessible = bank.accessible_floors
        actions = []
        for i, lift in enumerate(bank.lifts):
            ha = _heuristic_action(bank, lift, accessible)
            if ha is not None:
                actions.append(ha)
                continue

            if random.random() < self.epsilon:
                actions.append(random.randint(0, ACTION_DIM - 1))
            else:
                actions.append(int(np.argmax(q_vals)))

        self.last_actions = actions
        return actions

    def store_reward(self, reward, bank, tick, done=False):
        if self.last_state_key is None:
            return
        next_key = self._discretize(bank, tick)
        if next_key not in self.q_table:
            self.q_table[next_key] = np.zeros(ACTION_DIM)

        old_q = self.q_table[self.last_state_key]
        next_q = self.q_table[next_key]

        if self.last_actions:
            dominant = Counter(self.last_actions).most_common(1)[0][0]
            old_q[dominant] += self.alpha * (
                reward + GAMMA * np.max(next_q) * (1 - done) - old_q[dominant])

        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


class CMAESController:

    def __init__(self, bank_id, pop_size=12):
        self.bank_id = bank_id
        self.pop_size = pop_size
        self.param_dim = 10 * ACTION_DIM
        self.mean = np.zeros(self.param_dim)
        self.sigma = 0.5
        self.population = []
        self.fitnesses = []
        self.current_params = self.mean.copy()
        self.eval_idx = 0
        self.generation = 0
        self.best_params = self.mean.copy()
        self.best_fitness = -float('inf')
        self._generate_population()
        self._run_reward = 0.0

    def _generate_population(self):
        self.population = []
        self.fitnesses = []
        for _ in range(self.pop_size):
            noise = np.random.randn(self.param_dim) * self.sigma
            self.population.append(self.mean + noise)
            self.fitnesses.append(0.0)
        self.eval_idx = 0
        self.current_params = self.population[0]

    def _extract_features(self, bank, tick):
        total_waiting = bank.get_total_waiting()
        avg_floor = sum(l.floor for l in bank.lifts) / NUM_LIFTS_PER_BANK
        total_load = sum(l.load for l in bank.lifts)
        idle_count = sum(1 for l in bank.lifts if l.state == LiftState.IDLE)
        moving_up = sum(1 for l in bank.lifts if l.direction == Direction.UP)
        moving_down = sum(1 for l in bank.lifts if l.direction == Direction.DOWN)
        max_wait = max((bank.get_max_wait(f, tick) for f in range(NUM_FLOORS)), default=0)
        hour = (tick % 86400) / 3600.0
        features = np.array([
            total_waiting / 50.0,
            avg_floor / NUM_FLOORS,
            total_load / (NUM_LIFTS_PER_BANK * 15),
            idle_count / NUM_LIFTS_PER_BANK,
            moving_up / NUM_LIFTS_PER_BANK,
            moving_down / NUM_LIFTS_PER_BANK,
            max_wait / 120.0,
            hour / 24.0,
            bank.get_waiting_count(FLOOR_INDEX.get('G', 8)) / 20.0,
            bank.get_waiting_count(FLOOR_INDEX.get('LG', 7)) / 20.0,
        ], dtype=np.float32)
        return np.clip(features, 0, 1)

    def select_action(self, bank, tick):
        features = self._extract_features(bank, tick)
        W = self.current_params.reshape(ACTION_DIM, 10)
        action_scores = W @ features

        accessible = bank.accessible_floors
        actions = []
        for i, lift in enumerate(bank.lifts):
            ha = _heuristic_action(bank, lift, accessible)
            if ha is not None:
                actions.append(ha)
                continue
            actions.append(int(np.argmax(action_scores)))

        return actions

    def store_reward(self, reward, done=False):
        self._run_reward += reward

    def end_run(self):
        self.fitnesses[self.eval_idx] = self._run_reward
        self._run_reward = 0.0
        self.eval_idx += 1
        if self.eval_idx >= self.pop_size:
            self._evolve()
        else:
            self.current_params = self.population[self.eval_idx]

    def update(self):
        return 0.0

    def _evolve(self):
        ranked = sorted(range(self.pop_size), key=lambda i: -self.fitnesses[i])
        elite_size = max(self.pop_size // 3, 2)
        elite = [self.population[i] for i in ranked[:elite_size]]

        if self.fitnesses[ranked[0]] > self.best_fitness:
            self.best_fitness = self.fitnesses[ranked[0]]
            self.best_params = self.population[ranked[0]].copy()

        self.mean = np.mean(elite, axis=0)
        self.sigma *= 0.95 if self.generation > 5 else 1.0
        self.sigma = max(self.sigma, 0.01)

        self.generation += 1
        self._generate_population()

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.save(path + '_params', self.best_params)
        np.save(path + '_mean', self.mean)
        state = {
            'sigma': self.sigma,
            'generation': self.generation,
            'best_fitness': self.best_fitness,
            'eval_idx': self.eval_idx,
        }
        np.save(path + '_state', np.array([state], dtype=object))

    def load(self, path):
        params_path = path + '_params.npy'
        mean_path = path + '_mean.npy'
        state_path = path + '_state.npy'
        legacy_path = path if path.endswith('.npy') else path + '.npy'

        if os.path.exists(params_path):
            self.best_params = np.load(params_path)
            self.mean = np.load(mean_path) if os.path.exists(mean_path) else self.best_params.copy()
            if os.path.exists(state_path):
                state = np.load(state_path, allow_pickle=True)[0]
                self.sigma = state.get('sigma', self.sigma)
                self.generation = state.get('generation', 0)
                self.best_fitness = state.get('best_fitness', -float('inf'))
            self.current_params = self.best_params.copy()
            self._generate_population()
        elif os.path.exists(legacy_path):
            self.best_params = np.load(legacy_path)
            self.mean = self.best_params.copy()
            self.current_params = self.best_params.copy()
            self._generate_population()



class RoundRobinController:

    def __init__(self):
        self._targets = {}
        self._rr_idx = 0

    def get_actions(self, bank, tick):
        actions = [int(LiftAction.IDLE)] * NUM_LIFTS_PER_BANK
        accessible = bank.accessible_floors

        for i, lift in enumerate(bank.lifts):
            ha = _heuristic_action(bank, lift, accessible)
            if ha is not None:
                actions[i] = ha
                self._targets.pop(i, None)
                continue

            if i in self._targets:
                target = self._targets[i]
                waiting = [p for p in bank.hall_calls[target] if p.destination in accessible]
                if not waiting:
                    del self._targets[i]
                elif target > lift.floor:
                    actions[i] = int(LiftAction.MOVE_UP)
                    continue
                elif target < lift.floor:
                    actions[i] = int(LiftAction.MOVE_DOWN)
                    continue
                else:
                    actions[i] = int(LiftAction.STOP_OPEN)
                    del self._targets[i]
                    continue

        call_floors = []
        for f in range(NUM_FLOORS):
            if f not in accessible:
                continue
            waiting = [p for p in bank.hall_calls[f] if p.destination in accessible]
            if waiting:
                call_floors.append((f, len(waiting)))
        call_floors.sort(key=lambda x: -x[1])

        idle_lifts = [i for i in range(NUM_LIFTS_PER_BANK)
                      if i not in self._targets
                      and actions[i] == int(LiftAction.IDLE)
                      and not bank.lifts[i].destinations]

        for floor, count in call_floors:
            if not idle_lifts:
                break
            assigned_here = sum(1 for t in self._targets.values() if t == floor)
            needed = max(1, (count + 14) // 15) - assigned_here
            for _ in range(max(0, needed)):
                if not idle_lifts:
                    break
                li = idle_lifts.pop(0)
                self._targets[li] = floor
                if floor > bank.lifts[li].floor:
                    actions[li] = int(LiftAction.MOVE_UP)
                elif floor < bank.lifts[li].floor:
                    actions[li] = int(LiftAction.MOVE_DOWN)

        return actions


class IdleWaitController:

    def __init__(self, bank_id):
        if bank_id == 'A':
            floors = sorted(BANK_A_FLOOR_INDICES)
        else:
            floors = sorted(BANK_B_FLOOR_INDICES)
        step = max(1, len(floors) // NUM_LIFTS_PER_BANK)
        self.home_floors = [floors[min(i * step, len(floors) - 1)]
                            for i in range(NUM_LIFTS_PER_BANK)]

    def get_actions(self, bank, tick):
        actions = [int(LiftAction.IDLE)] * NUM_LIFTS_PER_BANK
        accessible = bank.accessible_floors

        for i, lift in enumerate(bank.lifts):
            ha = _heuristic_action(bank, lift, accessible)
            if ha is not None:
                actions[i] = ha
                continue

            best = -1
            best_d = NUM_FLOORS + 1
            for f in range(NUM_FLOORS):
                if f not in accessible:
                    continue
                waiting = [p for p in bank.hall_calls[f] if p.destination in accessible]
                if waiting:
                    d = abs(f - lift.floor)
                    if d < best_d:
                        best_d = d
                        best = f

            if best >= 0:
                if best > lift.floor:
                    actions[i] = int(LiftAction.MOVE_UP)
                elif best < lift.floor:
                    actions[i] = int(LiftAction.MOVE_DOWN)
            else:
                home = self.home_floors[i]
                if home > lift.floor:
                    actions[i] = int(LiftAction.MOVE_UP)
                elif home < lift.floor:
                    actions[i] = int(LiftAction.MOVE_DOWN)

        return actions


class AStarScanPPOController:

    def __init__(self, bank_id):
        self.bank_id = bank_id
        self.base = AStarScanController()
        self.model = ActorCritic(STATE_DIM, ACTION_DIM, NUM_LIFTS_PER_BANK)
        self.optimizer = optim.Adam(self.model.parameters(), lr=LEARNING_RATE)
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []
        self.reward_normalizer = RunningRewardNormalizer()

    def select_action(self, state, bank, tick):
        base_actions = self.base.get_actions(bank, tick)

        with torch.no_grad():
            s = torch.FloatTensor(state).unsqueeze(0)
            logits_list, value = self.model(s)

        raw_actions = []
        for logits in logits_list:
            dist = Categorical(logits=logits.squeeze(0))
            action = dist.sample()
            raw_actions.append(action.item())

        final = list(base_actions)
        for i, lift in enumerate(bank.lifts):
            if (base_actions[i] == int(LiftAction.IDLE)
                    and not lift.destinations
                    and lift.state == LiftState.IDLE):
                pa = raw_actions[i]
                if pa in (int(LiftAction.MOVE_UP), int(LiftAction.MOVE_DOWN)):
                    final[i] = pa

        # Compute log_probs for the EXECUTED actions (not raw NN outputs)
        log_probs = []
        for i, logits in enumerate(logits_list):
            dist = Categorical(logits=logits.squeeze(0))
            log_probs.append(dist.log_prob(torch.tensor(final[i])).item())

        self.states.append(state)
        self.actions.append(final)
        self.log_probs.append(log_probs)
        self.values.append(value.item())
        return final

    def store_reward(self, reward, done=False):
        self.rewards.append(reward)
        self.dones.append(done)

    def update(self):
        if len(self.states) < 32:
            return 0.0

        states = torch.FloatTensor(np.array(self.states))
        actions = torch.LongTensor(np.array(self.actions))
        old_log_probs = torch.FloatTensor(np.array(self.log_probs)).sum(dim=-1)
        raw_rewards = np.array(self.rewards)
        rewards = np.array([self.reward_normalizer.normalize(r) for r in raw_rewards])
        values = np.array(self.values)
        dones = np.array(self.dones, dtype=np.float32)

        advantages = np.zeros_like(rewards)
        last_gae = 0
        for t in reversed(range(len(rewards))):
            next_val = values[t + 1] if t < len(rewards) - 1 else 0
            delta = rewards[t] + GAMMA * next_val * (1 - dones[t]) - values[t]
            last_gae = delta + GAMMA * GAE_LAMBDA * (1 - dones[t]) * last_gae
            advantages[t] = last_gae

        returns = advantages + values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        advantages = torch.FloatTensor(advantages)
        returns = torch.FloatTensor(returns)

        total_loss = 0
        for _ in range(PPO_EPOCHS):
            indices = np.random.permutation(len(states))
            for start in range(0, len(states), BATCH_SIZE):
                idx = indices[start:start + BATCH_SIZE]
                new_lp, new_values, entropy = self.model.evaluate(
                    states[idx], actions[idx])
                ratio = torch.exp(new_lp - old_log_probs[idx])
                surr1 = ratio * advantages[idx]
                surr2 = torch.clamp(ratio, 1 - PPO_CLIP, 1 + PPO_CLIP) * advantages[idx]
                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = 0.5 * (new_values - returns[idx]).pow(2).mean()
                loss = actor_loss + critic_loss - 0.05 * entropy.mean()
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                self.optimizer.step()
                total_loss += loss.item()

        self.states.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.rewards.clear()
        self.values.clear()
        self.dones.clear()
        return total_loss

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        if os.path.exists(path):
            try:
                self.model.load_state_dict(torch.load(path, weights_only=True))
            except (RuntimeError, KeyError):
                pass


class QRDQNetwork(nn.Module):

    N_QUANTILES = 32

    def __init__(self, state_dim, action_dim, num_lifts):
        super().__init__()
        self.num_lifts = num_lifts
        self.action_dim = action_dim
        self.n_q = self.N_QUANTILES

        self.shared = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
        )
        self.q_heads = nn.ModuleList([
            nn.Linear(128, action_dim * self.n_q) for _ in range(num_lifts)
        ])

    def forward(self, state):
        features = self.shared(state)
        quantiles = []
        for head in self.q_heads:
            q = head(features)
            q = q.view(q.shape[0], self.action_dim, self.n_q)
            quantiles.append(q)
        return quantiles

    def get_q_values(self, state):
        quantiles = self(state)
        return [q.mean(dim=-1) for q in quantiles]


class QRDQNAgent:

    def __init__(self, bank_id):
        self.bank_id = bank_id
        self.q_net = QRDQNetwork(STATE_DIM, ACTION_DIM, NUM_LIFTS_PER_BANK)
        self.target_net = QRDQNetwork(STATE_DIM, ACTION_DIM, NUM_LIFTS_PER_BANK)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=LEARNING_RATE)

        self.replay_buffer = deque(maxlen=50000)
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.9995
        self.target_update_freq = 200
        self.step_count = 0
        self.last_state = None
        self.last_actions = None
        self.n_q = QRDQNetwork.N_QUANTILES

        self.tau = torch.FloatTensor(
            [(2 * i + 1) / (2 * self.n_q) for i in range(self.n_q)]
        )

    def select_action(self, state, bank=None):
        self.last_state = state
        accessible = bank.accessible_floors if bank else set()

        if random.random() < self.epsilon:
            raw_actions = [random.randint(0, ACTION_DIM - 1) for _ in range(NUM_LIFTS_PER_BANK)]
        else:
            with torch.no_grad():
                s = torch.FloatTensor(state).unsqueeze(0)
                q_values = self.q_net.get_q_values(s)
                raw_actions = [qv.argmax(dim=-1).item() for qv in q_values]

        if bank:
            actions = []
            for i, lift in enumerate(bank.lifts):
                if lift.state in (LiftState.DOOR_OPEN, LiftState.DECELERATING):
                    actions.append(int(LiftAction.IDLE))
                elif lift.floor in lift.destinations:
                    actions.append(int(LiftAction.STOP_OPEN))
                elif lift.passengers and lift.destinations:
                    nearest = min(lift.destinations, key=lambda d: abs(d - lift.floor))
                    if nearest > lift.floor:
                        actions.append(int(LiftAction.MOVE_UP))
                    else:
                        actions.append(int(LiftAction.MOVE_DOWN))
                else:
                    actions.append(raw_actions[i])
        else:
            actions = raw_actions

        self.last_actions = actions
        return actions

    def store_reward(self, reward, next_state=None, done=False):
        if self.last_state is not None and next_state is not None:
            self.replay_buffer.append(
                (self.last_state, self.last_actions, reward, next_state, done))

    def update(self):
        if len(self.replay_buffer) < BATCH_SIZE:
            return 0.0

        batch = random.sample(list(self.replay_buffer), BATCH_SIZE)
        states = torch.FloatTensor(np.array([b[0] for b in batch]))
        actions = [b[1] for b in batch]
        rewards = torch.FloatTensor([b[2] for b in batch])
        next_states = torch.FloatTensor(np.array([b[3] for b in batch]))
        dones = torch.FloatTensor([b[4] for b in batch])

        current_quantiles = self.q_net(states)
        with torch.no_grad():
            next_q_mean = self.target_net.get_q_values(next_states)
            next_quantiles = self.target_net(next_states)

        total_loss = torch.tensor(0.0)

        for lift_i in range(NUM_LIFTS_PER_BANK):
            a = torch.LongTensor([act[lift_i] for act in actions])
            cur_q = current_quantiles[lift_i]
            cur_q = cur_q[torch.arange(BATCH_SIZE), a]

            best_next_a = next_q_mean[lift_i].argmax(dim=-1)
            next_q = next_quantiles[lift_i]
            next_q = next_q[torch.arange(BATCH_SIZE), best_next_a]

            target = rewards.unsqueeze(1) + GAMMA * next_q * (1 - dones.unsqueeze(1))

            diff = target.unsqueeze(2) - cur_q.unsqueeze(1)
            huber = torch.where(diff.abs() < 1.0, 0.5 * diff.pow(2), diff.abs() - 0.5)
            tau = self.tau.view(1, 1, -1)
            qr_loss = (tau - (diff < 0).float()).abs() * huber
            total_loss = total_loss + qr_loss.mean()

        self.optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), 1.0)
        self.optimizer.step()

        self.step_count += 1
        if self.step_count % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        return total_loss.item()

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'model': self.q_net.state_dict(),
            'epsilon': self.epsilon,
            'step_count': self.step_count,
        }, path)

    def load(self, path):
        if os.path.exists(path):
            try:
                ckpt = torch.load(path, weights_only=False)
                if isinstance(ckpt, dict) and 'model' in ckpt:
                    self.q_net.load_state_dict(ckpt['model'])
                    self.epsilon = ckpt.get('epsilon', self.epsilon_min)
                    self.step_count = ckpt.get('step_count', 0)
                else:
                    self.q_net.load_state_dict(ckpt)
                self.target_net.load_state_dict(self.q_net.state_dict())
            except (RuntimeError, KeyError):
                pass
