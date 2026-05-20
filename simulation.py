import threading
import time
import logging
import os
from typing import Optional, List

from environment import BuildingEnvironment
from agents import (PPOAgent, ScanController, NearestFirstController,
                    AStarDispatchController, AStarScanController,
                    DQNAgent, REINFORCEAgent,
                    QLearningController, CMAESController,
                    RoundRobinController, IdleWaitController,
                    AStarScanPPOController, QRDQNAgent)
from traffic import TrafficGenerator
from database import create_run, save_tick, finish_run, save_training_log, save_run_summary
from config import FLOOR_NAMES, NUM_FLOORS


RL_ALGORITHMS = {'ppo', 'dqn', 'reinforce', 'qlearning', 'cmaes',
                 'astar_scan_ppo', 'qrdqn'}
TRAINABLE_ALGORITHMS = {'ppo', 'dqn', 'reinforce', 'cmaes',
                        'astar_scan_ppo', 'qrdqn'}

# ── Training run logger ────────────────────────────────────────────────────────
os.makedirs('logs', exist_ok=True)
_train_logger = logging.getLogger('autotrain')
if not _train_logger.handlers:
    _train_logger.setLevel(logging.INFO)
    _fh = logging.FileHandler('logs/auto_train.log', encoding='utf-8')
    _fh.setFormatter(logging.Formatter('%(asctime)s  %(message)s',
                                       datefmt='%Y-%m-%d %H:%M:%S'))
    _ch = logging.StreamHandler()           # also prints to console
    _ch.setFormatter(logging.Formatter('%(asctime)s  %(message)s',
                                       datefmt='%H:%M:%S'))
    _train_logger.addHandler(_fh)
    _train_logger.addHandler(_ch)
    _train_logger.propagate = False
# ──────────────────────────────────────────────────────────────────────────────


class SimulationRunner:

    def __init__(self):
        self.env: Optional[BuildingEnvironment] = None
        self.traffic: Optional[TrafficGenerator] = None
        self.algorithm = 'scan'
        self.running = False
        self.paused = False
        self.run_id: Optional[int] = None
        self.speed = 1
        self.thread: Optional[threading.Thread] = None
        self.max_ticks = 3600
        self.sim_start_time = 9 * 3600
        self.save_interval = 10

        self.ppo_agent_a: Optional[PPOAgent] = None
        self.ppo_agent_b: Optional[PPOAgent] = None
        self.scan_a = None
        self.scan_b = None
        self.nearest_a = None
        self.nearest_b = None
        self.astar_a = None
        self.astar_b = None
        self.astar_scan_a = None
        self.astar_scan_b = None
        self.dqn_a: Optional[DQNAgent] = None
        self.dqn_b: Optional[DQNAgent] = None
        self.reinforce_a: Optional[REINFORCEAgent] = None
        self.reinforce_b: Optional[REINFORCEAgent] = None
        self.qlearn_a = None
        self.qlearn_b = None
        self.cmaes_a: Optional[CMAESController] = None
        self.cmaes_b: Optional[CMAESController] = None
        self.roundrobin_a = None
        self.roundrobin_b = None
        self.idlewait_a = None
        self.idlewait_b = None
        self.asp_a: Optional[AStarScanPPOController] = None
        self.asp_b: Optional[AStarScanPPOController] = None
        self.qrdqn_a: Optional[QRDQNAgent] = None
        self.qrdqn_b: Optional[QRDQNAgent] = None

        self.latest_info = {}
        self.cumulative_reward_a = 0
        self.cumulative_reward_b = 0
        self.cumulative_reward = 0

        # Per-run peak/aggregate stats (reset each episode)
        self._peak_concurrent_waiting = 0
        self._peak_max_wait = 0.0
        self._avg_wait_samples: List[float] = []

        self._last_actions_a: List[int] = [3] * 8
        self._last_actions_b: List[int] = [3] * 8
        self._action_repeat = 4

        self.auto_train_running = False
        self.auto_train_current = 0
        self.auto_train_total = 0
        self.auto_train_config: dict = {}
        self.auto_train_thread: Optional[threading.Thread] = None
        self.auto_train_results: List[dict] = []
        self.auto_train_summary: dict = {}

    def start(self, name, algorithm, duration_minutes=60,
              start_hour=9.0, speed=1,
              custom_arrival_rate=None, train_ppo=False,
              _reuse_agents=False):
        if self.running:
            self.stop()

        self.env = BuildingEnvironment()
        self.algorithm = algorithm
        self.speed = speed
        self.max_ticks = duration_minutes * 60
        self.sim_start_time = int(start_hour * 3600)
        self.cumulative_reward_a = 0
        self.cumulative_reward_b = 0
        self.cumulative_reward = 0
        self._peak_concurrent_waiting = 0
        self._peak_max_wait = 0.0
        self._avg_wait_samples = []

        if speed < 100:
            self.save_interval = 10
        elif speed < 500:
            self.save_interval = 30
        elif speed < 2000:
            self.save_interval = 100
        else:
            self.save_interval = 300

        self.traffic = TrafficGenerator(
            custom_rate_override=custom_arrival_rate
        )

        # Disable background urgency hijacking for baselines; enable for all others.
        if algorithm in ('scan', 'nearest', 'roundrobin', 'idlewait'):
            self.env.urgency_dispatch_enabled = False
        else:
            self.env.urgency_dispatch_enabled = True

        if algorithm == 'ppo':
            if not (_reuse_agents and self.ppo_agent_a):
                self.ppo_agent_a = PPOAgent('A')
                self.ppo_agent_b = PPOAgent('B')
                self.ppo_agent_a.load('models/ppo_bank_a.pt')
                self.ppo_agent_b.load('models/ppo_bank_b.pt')
        elif algorithm == 'scan':
            self.scan_a = ScanController(self.env.bank_a)
            self.scan_b = ScanController(self.env.bank_b)
        elif algorithm == 'nearest':
            self.nearest_a = NearestFirstController()
            self.nearest_b = NearestFirstController()
        elif algorithm == 'astar':
            self.astar_a = AStarDispatchController()
            self.astar_b = AStarDispatchController()
        elif algorithm == 'astar_scan':
            self.astar_scan_a = AStarScanController()
            self.astar_scan_b = AStarScanController()
        elif algorithm == 'dqn':
            if not (_reuse_agents and self.dqn_a):
                self.dqn_a = DQNAgent('A')
                self.dqn_b = DQNAgent('B')
                self.dqn_a.load('models/dqn_bank_a.pt')
                self.dqn_b.load('models/dqn_bank_b.pt')
        elif algorithm == 'reinforce':
            if not (_reuse_agents and self.reinforce_a):
                self.reinforce_a = REINFORCEAgent('A')
                self.reinforce_b = REINFORCEAgent('B')
                self.reinforce_a.load('models/reinforce_bank_a.pt')
                self.reinforce_b.load('models/reinforce_bank_b.pt')
        elif algorithm == 'qlearning':
            if not (_reuse_agents and self.qlearn_a):
                self.qlearn_a = QLearningController('A')
                self.qlearn_b = QLearningController('B')
        elif algorithm == 'cmaes':
            if not (_reuse_agents and self.cmaes_a):
                self.cmaes_a = CMAESController('A')
                self.cmaes_b = CMAESController('B')
                
                # ─────────────────────────────────────────────────────────────────
                # MARATHON PROTECTION: If training is OFF, load the Elite model.
                # If training is ON, load the latest evolutionary state.
                # ─────────────────────────────────────────────────────────────────
                if not train_ppo and os.path.exists('models/best_cmaes_bank_a.npy'):
                    self.cmaes_a.load('models/best_cmaes_bank_a')
                    self.cmaes_b.load('models/best_cmaes_bank_b')
                else:
                    self.cmaes_a.load('models/cmaes_bank_a')
                    self.cmaes_b.load('models/cmaes_bank_b')
        elif algorithm == 'roundrobin':
            self.roundrobin_a = RoundRobinController()
            self.roundrobin_b = RoundRobinController()
        elif algorithm == 'idlewait':
            self.idlewait_a = IdleWaitController('A')
            self.idlewait_b = IdleWaitController('B')
        elif algorithm == 'astar_scan_ppo':
            if not (_reuse_agents and self.asp_a):
                self.asp_a = AStarScanPPOController('A')
                self.asp_b = AStarScanPPOController('B')
                self.asp_a.load('models/asp_bank_a.pt')
                self.asp_b.load('models/asp_bank_b.pt')
        elif algorithm == 'qrdqn':
            if not (_reuse_agents and self.qrdqn_a):
                self.qrdqn_a = QRDQNAgent('A')
                self.qrdqn_b = QRDQNAgent('B')
                self.qrdqn_a.load('models/qrdqn_bank_a.pt')
                self.qrdqn_b.load('models/qrdqn_bank_b.pt')

        self.run_id = create_run(
            name=name, algorithm=algorithm,
            sim_start_hour=start_hour, speed_multiplier=speed,
            custom_arrival_rate=custom_arrival_rate,
            config={'duration_minutes': duration_minutes, 'train_ppo': train_ppo}
        )

        self.running = True
        self.paused = False
        self.train_ppo = train_ppo

        self.thread = threading.Thread(target=self._run_loop, daemon=True)
        self.thread.start()
        return self.run_id

    def _run_loop(self):
        while self.running and self.env.tick < self.max_ticks:
            if self.paused:
                time.sleep(0.1)
                continue

            sim_time = self.sim_start_time + self.env.tick

            arrivals = self.traffic.generate_arrivals(sim_time)
            for origin, dest in arrivals:
                self.env.add_passenger(origin, dest)

            actions_a, actions_b = self._get_actions()
            self._last_actions_a = actions_a
            self._last_actions_b = actions_b

            reward_a, reward_b, safe_a, safe_b = self.env.step(actions_a, actions_b)
            # Update DQN stored actions to match the safety-filtered executed actions
            if self.algorithm == 'dqn':
                self.dqn_a.last_actions = safe_a
                self.dqn_b.last_actions = safe_b
            elif self.algorithm == 'qrdqn':
                self.qrdqn_a.last_actions = safe_a
                self.qrdqn_b.last_actions = safe_b
            self.cumulative_reward_a += reward_a
            self.cumulative_reward_b += reward_b
            self.cumulative_reward += reward_a + reward_b

            prev = self.latest_info
            self.latest_info = {
                'tick': self.env.tick,
                'total_waiting': (self.env.bank_a.get_total_waiting()
                                  + self.env.bank_b.get_total_waiting()),
                'total_delivered': len(self.env.delivered_passengers),
                'total_energy': self.env.total_energy,
                'tick_energy': self.env.tick_energy,
                'avg_wait':          prev.get('avg_wait', 0),
                'max_wait':          prev.get('max_wait', 0),
                'p95_wait':          prev.get('p95_wait', 0),
                'throughput_per_min': prev.get('throughput_per_min', 0),
                'bank_a_lifts':      prev.get('bank_a_lifts', []),
                'bank_b_lifts':      prev.get('bank_b_lifts', []),
                'floor_waiting':     prev.get('floor_waiting', {}),
            }

            # Track per-run peak/aggregate stats for end-of-episode log
            tw = self.latest_info['total_waiting']
            mw = self.latest_info['max_wait']
            aw = self.latest_info['avg_wait']
            if tw > self._peak_concurrent_waiting:
                self._peak_concurrent_waiting = tw
            if mw > self._peak_max_wait:
                self._peak_max_wait = mw
            if aw > 0:
                self._avg_wait_samples.append(aw)

            self._train_step(reward_a, reward_b)

            if self.env.tick % self.save_interval == 0:
                full_info = self.env.get_info()
                # Skip disk I/O during Auto-Train — only final summary is needed.
                if not self.auto_train_running:
                    save_tick(self.run_id, full_info)
                self.latest_info.update(full_info)

            if self.speed < 100:
                time.sleep(1.0 / max(self.speed, 1))

        self._finish_run()
        self.running = False

    def _train_step(self, reward_a, reward_b):
        algo = self.algorithm
        tick = self.env.tick

        if algo == 'ppo':
            self.ppo_agent_a.store_reward(reward_a)
            self.ppo_agent_b.store_reward(reward_b)
            if self.train_ppo and tick % 256 == 0 and tick > 0:
                loss_a = self.ppo_agent_a.update()
                loss_b = self.ppo_agent_b.update()
                self._log_training(tick // 256, loss_a + loss_b)

        elif algo == 'dqn':
            cur_a = self.env.get_per_lift_rewards(self.env.bank_a)
            cur_b = self.env.get_per_lift_rewards(self.env.bank_b)
            ns_a = self.env.get_state('A')
            ns_b = self.env.get_state('B')
            self.dqn_a.store_reward(cur_a, ns_a, bank=self.env.bank_a)
            self.dqn_b.store_reward(cur_b, ns_b, bank=self.env.bank_b)
            if self.train_ppo and tick % 64 == 0 and tick > 0:
                loss_a = self.dqn_a.update()
                loss_b = self.dqn_b.update()
                if tick % 256 == 0:
                    self._log_training(tick // 256, loss_a + loss_b)

        elif algo == 'reinforce':
            self.reinforce_a.store_reward(reward_a)
            self.reinforce_b.store_reward(reward_b)
            if self.train_ppo and tick % 256 == 0 and tick > 0:
                loss_a = self.reinforce_a.update()
                loss_b = self.reinforce_b.update()
                self._log_training(tick // 256, (loss_a or 0) + (loss_b or 0))

        elif algo == 'qlearning':
            self.qlearn_a.store_reward(reward_a, self.env.bank_a,
                                       self.sim_start_time + tick)
            self.qlearn_b.store_reward(reward_b, self.env.bank_b,
                                       self.sim_start_time + tick)

        elif algo == 'cmaes':
            self.cmaes_a.store_reward(reward_a)
            self.cmaes_b.store_reward(reward_b)

        elif algo == 'astar_scan_ppo':
            self.asp_a.store_reward(reward_a)
            self.asp_b.store_reward(reward_b)
            if self.train_ppo and tick % 256 == 0 and tick > 0:
                loss_a = self.asp_a.update()
                loss_b = self.asp_b.update()
                self._log_training(tick // 256, loss_a + loss_b)

        elif algo == 'qrdqn':
            ns_a = self.env.get_state('A')
            ns_b = self.env.get_state('B')
            self.qrdqn_a.store_reward(reward_a, ns_a)
            self.qrdqn_b.store_reward(reward_b, ns_b)
            if self.train_ppo and tick % 64 == 0 and tick > 0:
                loss_a = self.qrdqn_a.update()
                loss_b = self.qrdqn_b.update()
                if tick % 256 == 0:
                    self._log_training(tick // 256, loss_a + loss_b)

    def _log_training(self, episode, loss):
        save_training_log(
            self.run_id, episode,
            self.cumulative_reward_a, self.cumulative_reward_b,
            self.latest_info.get('avg_wait', 0),
            self.latest_info.get('total_delivered', 0),
            self.env.total_energy, loss
        )

    def _finish_run(self):
        if not self.run_id:
            return
        finish_run(self.run_id, self.env.tick)

        # Always save a final tick-stats row so report summaries are never all-zero
        # (during auto-train the periodic save_tick calls are skipped for speed).
        final_info = self.env.get_info()
        save_tick(self.run_id, final_info)
        self.latest_info.update(final_info)

        # Persist accurate in-memory aggregates so the report matches the dashboard.
        # tick_stats averages are distorted when only one row was saved (auto-train).
        mean_aw = (sum(self._avg_wait_samples) / len(self._avg_wait_samples)
                   if self._avg_wait_samples else 0.0)
        save_run_summary(self.run_id, {
            'mean_avg_wait':   round(mean_aw, 2),
            'peak_max_wait':   round(self._peak_max_wait, 2),
            'peak_waiting':    self._peak_concurrent_waiting,
            'total_reward':    round(self.cumulative_reward, 1),
        })

        algo = self.algorithm

        # Run the single end-of-episode optimisation for trajectory-based
        # algorithms now that the simulation loop has fully finished.
        # DQN / QR-DQN update inline (replay buffer) so they are excluded.
        if self.train_ppo:
            if algo == 'ppo' and self.ppo_agent_a:
                _train_logger.info('Run complete — optimising PPO networks...')
                loss_a = self.ppo_agent_a.update() or 0
                loss_b = self.ppo_agent_b.update() or 0
                self._log_training(self.env.tick // self._action_repeat, loss_a + loss_b)
            elif algo == 'reinforce' and self.reinforce_a:
                _train_logger.info('Run complete — optimising REINFORCE networks...')
                loss_a = self.reinforce_a.update() or 0
                loss_b = self.reinforce_b.update() or 0
                self._log_training(self.env.tick // self._action_repeat, (loss_a or 0) + (loss_b or 0))
            elif algo == 'astar_scan_ppo' and self.asp_a:
                _train_logger.info('Run complete — optimising ASP networks...')
                loss_a = self.asp_a.update() or 0
                loss_b = self.asp_b.update() or 0
                self._log_training(self.env.tick // self._action_repeat, loss_a + loss_b)

        if algo == 'ppo' and self.ppo_agent_a:
            self.ppo_agent_a.save('models/ppo_bank_a.pt')
            self.ppo_agent_b.save('models/ppo_bank_b.pt')
        elif algo == 'dqn' and self.dqn_a:
            self.dqn_a.save('models/dqn_bank_a.pt')
            self.dqn_b.save('models/dqn_bank_b.pt')
        elif algo == 'reinforce' and self.reinforce_a:
            self.reinforce_a.save('models/reinforce_bank_a.pt')
            self.reinforce_b.save('models/reinforce_bank_b.pt')
        elif algo == 'cmaes' and self.cmaes_a:
            if self.train_ppo:
                self.cmaes_a.end_run()
                self.cmaes_b.end_run()
            # Always save the latest state so training can resume normally
            self.cmaes_a.save('models/cmaes_bank_a')
            self.cmaes_b.save('models/cmaes_bank_b')
            
            # ─────────────────────────────────────────────────────────────────
            # ELITE CHECKPOINTING: Save a locked copy of the highest scoring agent
            # ─────────────────────────────────────────────────────────────────
            if self.train_ppo:
                if not hasattr(self, 'best_training_reward'):
                    self.best_training_reward = -float('inf')
                
                if self.cumulative_reward > self.best_training_reward:
                    self.best_training_reward = self.cumulative_reward
                    self.cmaes_a.save('models/best_cmaes_bank_a')
                    self.cmaes_b.save('models/best_cmaes_bank_b')
                    _train_logger.info(f"*** NEW ELITE POLICY SAVED! Reward: {self.best_training_reward:.1f} ***")
        elif algo == 'astar_scan_ppo' and self.asp_a:
            self.asp_a.save('models/asp_bank_a.pt')
            self.asp_b.save('models/asp_bank_b.pt')
        elif algo == 'qrdqn' and self.qrdqn_a:
            self.qrdqn_a.save('models/qrdqn_bank_a.pt')
            self.qrdqn_b.save('models/qrdqn_bank_b.pt')

    def _get_actions(self):
        sim_time = self.sim_start_time + self.env.tick
        algo = self.algorithm

        if algo == 'ppo':
            state_a = self.env.get_state('A')
            state_b = self.env.get_state('B')
            actions_a = self.ppo_agent_a.select_action(state_a, self.env.bank_a)
            actions_b = self.ppo_agent_b.select_action(state_b, self.env.bank_b)
        elif algo == 'scan':
            actions_a = self.scan_a.get_actions(self.env.bank_a, self.env.tick)
            actions_b = self.scan_b.get_actions(self.env.bank_b, self.env.tick)
        elif algo == 'nearest':
            actions_a = self.nearest_a.get_actions(self.env.bank_a, self.env.tick)
            actions_b = self.nearest_b.get_actions(self.env.bank_b, self.env.tick)
        elif algo == 'astar':
            actions_a = self.astar_a.get_actions(self.env.bank_a, self.env.tick)
            actions_b = self.astar_b.get_actions(self.env.bank_b, self.env.tick)
        elif algo == 'astar_scan':
            actions_a = self.astar_scan_a.get_actions(self.env.bank_a, self.env.tick)
            actions_b = self.astar_scan_b.get_actions(self.env.bank_b, self.env.tick)
        elif algo == 'dqn':
            state_a = self.env.get_state('A')
            state_b = self.env.get_state('B')
            actions_a = self.dqn_a.select_action(state_a, self.env.bank_a)
            actions_b = self.dqn_b.select_action(state_b, self.env.bank_b)
        elif algo == 'reinforce':
            state_a = self.env.get_state('A')
            state_b = self.env.get_state('B')
            actions_a = self.reinforce_a.select_action(state_a, self.env.bank_a)
            actions_b = self.reinforce_b.select_action(state_b, self.env.bank_b)
        elif algo == 'qlearning':
            actions_a = self.qlearn_a.select_action(self.env.bank_a, sim_time)
            actions_b = self.qlearn_b.select_action(self.env.bank_b, sim_time)
        elif algo == 'cmaes':
            actions_a = self.cmaes_a.select_action(self.env.bank_a, sim_time)
            actions_b = self.cmaes_b.select_action(self.env.bank_b, sim_time)
        elif algo == 'roundrobin':
            actions_a = self.roundrobin_a.get_actions(self.env.bank_a, self.env.tick)
            actions_b = self.roundrobin_b.get_actions(self.env.bank_b, self.env.tick)
        elif algo == 'idlewait':
            actions_a = self.idlewait_a.get_actions(self.env.bank_a, self.env.tick)
            actions_b = self.idlewait_b.get_actions(self.env.bank_b, self.env.tick)
        elif algo == 'astar_scan_ppo':
            state_a = self.env.get_state('A')
            state_b = self.env.get_state('B')
            actions_a = self.asp_a.select_action(state_a, self.env.bank_a, self.env.tick)
            actions_b = self.asp_b.select_action(state_b, self.env.bank_b, self.env.tick)
        elif algo == 'qrdqn':
            state_a = self.env.get_state('A')
            state_b = self.env.get_state('B')
            actions_a = self.qrdqn_a.select_action(state_a, self.env.bank_a)
            actions_b = self.qrdqn_b.select_action(state_b, self.env.bank_b)
        else:
            actions_a = [3] * 8
            actions_b = [3] * 8
        return actions_a, actions_b

    def add_burst(self, floor_name, count, direction='random'):
        if not self.env:
            return
        floor_idx = FLOOR_NAMES.index(floor_name) if floor_name in FLOOR_NAMES else 8
        trips = self.traffic.generate_burst(floor_idx, count, direction)
        for origin, dest in trips:
            self.env.add_passenger(origin, dest)

    def pause(self):
        self.paused = True

    def resume(self):
        self.paused = False

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=5)

    def start_auto_train(self, config, n_runs):
        if self.auto_train_running:
            return {'error': 'Auto-train already running'}
        self.auto_train_running = True
        self.auto_train_current = 0
        self.auto_train_total = n_runs
        self.auto_train_config = config
        self.auto_train_thread = threading.Thread(
            target=self._auto_train_loop, daemon=True)
        self.auto_train_thread.start()
        return {'status': 'started', 'total': n_runs}

    def stop_auto_train(self):
        self.auto_train_running = False
        self.stop()

    def get_auto_train_status(self):
        return {
            'running': self.auto_train_running,
            'current': self.auto_train_current,
            'total': self.auto_train_total,
            'algorithm': self.auto_train_config.get('algorithm', ''),
            'sim_running': self.running,
            'has_summary': bool(self.auto_train_summary),
        }

    def _log_run_summary(self, run_name):
        """Write a one-line per-episode summary to logs/auto_train.log and stdout."""
        delivered   = self.latest_info.get('total_delivered', 0)
        energy      = self.latest_info.get('total_energy', 0)
        mean_aw     = (sum(self._avg_wait_samples) / len(self._avg_wait_samples)
                       if self._avg_wait_samples else 0.0)
        peak_mw     = self._peak_max_wait
        peak_cw     = self._peak_concurrent_waiting
        rew_a       = round(self.cumulative_reward_a, 1)
        rew_b       = round(self.cumulative_reward_b, 1)
        rew_total   = round(self.cumulative_reward, 1)
        algo        = self.algorithm.upper()

        # Assessment flags
        aw_flag  = 'GOOD' if mean_aw  < 30   else ('OK' if mean_aw  < 60   else 'POOR')
        mw_flag  = 'GOOD' if peak_mw  < 60   else ('OK' if peak_mw  < 120  else 'POOR')
        cw_flag  = 'GOOD' if peak_cw  < 20   else ('OK' if peak_cw  < 50   else 'POOR')

        _train_logger.info(
            '[%s] %-30s | Delivered: %4d | '
            'AvgWait: %6.0fs [%s] | PeakMaxWait: %6.0fs [%s] | '
            'PeakWaiting: %3d [%s] | Energy: %7.0f | Reward: %.0f/%.0f',
            algo, run_name,
            delivered,
            mean_aw,  aw_flag,
            peak_mw,  mw_flag,
            peak_cw,  cw_flag,
            energy,
            rew_a, rew_b,
        )

    def _auto_train_loop(self):
        cfg = self.auto_train_config
        algo = cfg.get('algorithm', 'ppo').upper()
        base = cfg.get('base_name', 'AutoTrain')
        self.auto_train_results = []
        self.auto_train_summary = {}
        _train_logger.info('='*90)
        _train_logger.info('AUTO-TRAIN START  algo=%s  runs=%d  dur=%dmin  speed=%dx',
                           algo, self.auto_train_total,
                           cfg.get('duration', 60), cfg.get('speed', 5000))
        _train_logger.info('='*90)
        _train_logger.info(
            '%-6s  %-30s  %8s  %10s  %14s  %13s  %8s  %14s',
            'Run', 'Name', 'Delivered',
            'AvgWait(s)', 'PeakMaxWait(s)', 'PeakWaiting',
            'Energy', 'Reward A/B')
        _train_logger.info('-'*110)

        for i in range(self.auto_train_total):
            if not self.auto_train_running:
                break
            self.auto_train_current = i + 1
            run_name = f"{base} #{self.auto_train_current}"

            # ─────────────────────────────────────────────────────────────────
            # GENERALIZATION FIX: Rotate the start hour across the day so the
            # 150-minute training window covers all traffic profiles.
            # ─────────────────────────────────────────────────────────────────
            start_hours = [8.0, 11.5, 16.5]  # 8:00 AM, 11:30 AM, 4:30 PM
            dynamic_start = start_hours[i % len(start_hours)]

            # If the user explicitly set a custom start_hour in the UI other
            # than the 8.0 default, respect it. Otherwise, use the rotation.
            ui_start = cfg.get('start_hour', 8.0)
            chosen_start = dynamic_start if ui_start == 8.0 else ui_start

            self.start(
                name=run_name,
                algorithm=cfg.get('algorithm', 'ppo'),
                duration_minutes=cfg.get('duration', 60),
                start_hour=chosen_start,
                speed=cfg.get('speed', 5000),
                train_ppo=True,
                _reuse_agents=(i > 0),
            )
            time.sleep(0.2)
            while self.running and self.auto_train_running:
                time.sleep(0.5)

            # Log end-of-episode summary
            self._log_run_summary(run_name)

            # Collect per-run stats for the final summary
            mean_aw = (sum(self._avg_wait_samples) / len(self._avg_wait_samples)
                       if self._avg_wait_samples else 0.0)
            self.auto_train_results.append({
                'run_num':   self.auto_train_current,
                'run_name':  run_name,
                'run_id':    self.run_id,
                'delivered': self.latest_info.get('total_delivered', 0),
                'mean_avg_wait': round(mean_aw, 1),
                'peak_max_wait': round(self._peak_max_wait, 1),
                'peak_concurrent': self._peak_concurrent_waiting,
                'energy':    round(self.latest_info.get('total_energy', 0), 0),
                'reward':    round(self.cumulative_reward, 1),
                'avg_journey_time': round(self.latest_info.get('avg_journey_time', 0), 1),
            })

        # Build cross-run summary
        completed = len(self.auto_train_results)
        if completed > 0:
            rs = self.auto_train_results
            self.auto_train_summary = {
                'algorithm':      algo,
                'total_runs':     self.auto_train_total,
                'completed_runs': completed,
                'runs':           rs,
                'best_delivered': max(rs, key=lambda r: r['delivered']),
                'worst_delivered': min(rs, key=lambda r: r['delivered']),
                'best_wait':      min(rs, key=lambda r: r['mean_avg_wait']),
                'worst_wait':     max(rs, key=lambda r: r['mean_avg_wait']),
                'averages': {
                    'delivered':       round(sum(r['delivered']       for r in rs) / completed, 1),
                    'mean_avg_wait':   round(sum(r['mean_avg_wait']   for r in rs) / completed, 1),
                    'peak_max_wait':   round(sum(r['peak_max_wait']   for r in rs) / completed, 1),
                    'peak_concurrent': round(sum(r['peak_concurrent'] for r in rs) / completed, 1),
                    'energy':          round(sum(r['energy']          for r in rs) / completed, 0),
                    'reward':          round(sum(r['reward']          for r in rs) / completed, 1),
                    'avg_journey_time': round(sum(r['avg_journey_time'] for r in rs) / completed, 1),
                },
            }

        _train_logger.info('='*90)
        _train_logger.info('AUTO-TRAIN COMPLETE  algo=%s  total_runs=%d', algo, self.auto_train_current)
        _train_logger.info('='*90)
        self.auto_train_running = False

    def get_live_state(self):
        if not self.env:
            return {}
        info = self.latest_info.copy() if self.latest_info else {}
        info['running'] = self.running
        info['paused'] = self.paused
        info['algorithm'] = self.algorithm
        info['cumulative_reward_a'] = round(self.cumulative_reward_a, 2)
        info['cumulative_reward_b'] = round(self.cumulative_reward_b, 2)
        info['cumulative_reward']   = round(self.cumulative_reward, 2)
        info['sim_time_seconds'] = self.sim_start_time + (self.env.tick if self.env else 0)
        info['active_passengers'] = len(self.env.active_passengers) if self.env else 0
        return info
