import threading
import time
from typing import Optional, List

from environment import BuildingEnvironment
from agents import (PPOAgent, ScanController, NearestFirstController,
                    AStarDispatchController, AStarScanController,
                    DQNAgent, REINFORCEAgent,
                    QLearningController, CMAESController,
                    RoundRobinController, IdleWaitController,
                    AStarScanPPOController, QRDQNAgent)
from traffic import TrafficGenerator
from database import create_run, save_tick, finish_run, save_training_log
from config import FLOOR_NAMES, NUM_FLOORS


RL_ALGORITHMS = {'ppo', 'dqn', 'reinforce', 'qlearning', 'cmaes',
                 'astar_scan_ppo', 'qrdqn'}
TRAINABLE_ALGORITHMS = {'ppo', 'dqn', 'reinforce', 'cmaes',
                        'astar_scan_ppo', 'qrdqn'}


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

        self._last_actions_a: List[int] = [3] * 8
        self._last_actions_b: List[int] = [3] * 8
        self._action_repeat = 4

        self.auto_train_running = False
        self.auto_train_current = 0
        self.auto_train_total = 0
        self.auto_train_config: dict = {}
        self.auto_train_thread: Optional[threading.Thread] = None

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

            reward_a, reward_b = self.env.step(actions_a, actions_b)
            self.cumulative_reward_a += reward_a
            self.cumulative_reward_b += reward_b

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

            self._train_step(reward_a, reward_b)

            if self.env.tick % self.save_interval == 0:
                full_info = self.env.get_info()
                save_tick(self.run_id, full_info)
                self.latest_info.update(full_info)

            if self.speed < 100:
                time.sleep(1.0 / max(self.speed, 1))

        self._finish_run()
        self.running = False

    def _train_step(self, reward_a, reward_b):
        algo = self.algorithm
        tick = self.env.tick
        is_decision_tick = (tick % self._action_repeat == 0 and tick > 0)

        if algo == 'ppo':
            self.ppo_agent_a._pending_reward = getattr(self.ppo_agent_a, '_pending_reward', 0) + reward_a
            self.ppo_agent_b._pending_reward = getattr(self.ppo_agent_b, '_pending_reward', 0) + reward_b
            if is_decision_tick:
                self.ppo_agent_a.store_reward(self.ppo_agent_a._pending_reward)
                self.ppo_agent_b.store_reward(self.ppo_agent_b._pending_reward)
                self.ppo_agent_a._pending_reward = 0.0
                self.ppo_agent_b._pending_reward = 0.0
            if self.train_ppo and tick % 256 == 0 and tick > 0:
                loss_a = self.ppo_agent_a.update()
                loss_b = self.ppo_agent_b.update()
                self._log_training(tick // 256, loss_a + loss_b)

        elif algo == 'dqn':
            self.dqn_a._pending_reward = getattr(self.dqn_a, '_pending_reward', 0) + reward_a
            self.dqn_b._pending_reward = getattr(self.dqn_b, '_pending_reward', 0) + reward_b
            if is_decision_tick:
                ns_a = self.env.get_state('A')
                ns_b = self.env.get_state('B')
                self.dqn_a.store_reward(self.dqn_a._pending_reward, ns_a)
                self.dqn_b.store_reward(self.dqn_b._pending_reward, ns_b)
                self.dqn_a._pending_reward = 0.0
                self.dqn_b._pending_reward = 0.0
            if self.train_ppo and tick % 16 == 0 and tick > 0:
                loss_a = self.dqn_a.update()
                loss_b = self.dqn_b.update()
                if tick % 256 == 0:
                    self._log_training(tick // 256, loss_a + loss_b)

        elif algo == 'reinforce':
            self.reinforce_a._pending_reward = getattr(self.reinforce_a, '_pending_reward', 0) + reward_a
            self.reinforce_b._pending_reward = getattr(self.reinforce_b, '_pending_reward', 0) + reward_b
            if is_decision_tick:
                self.reinforce_a.store_reward(self.reinforce_a._pending_reward)
                self.reinforce_b.store_reward(self.reinforce_b._pending_reward)
                self.reinforce_a._pending_reward = 0.0
                self.reinforce_b._pending_reward = 0.0
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
            self.asp_a._pending_reward = getattr(self.asp_a, '_pending_reward', 0) + reward_a
            self.asp_b._pending_reward = getattr(self.asp_b, '_pending_reward', 0) + reward_b
            if is_decision_tick:
                self.asp_a.store_reward(self.asp_a._pending_reward)
                self.asp_b.store_reward(self.asp_b._pending_reward)
                self.asp_a._pending_reward = 0.0
                self.asp_b._pending_reward = 0.0
            if self.train_ppo and tick % 256 == 0 and tick > 0:
                loss_a = self.asp_a.update()
                loss_b = self.asp_b.update()
                self._log_training(tick // 256, loss_a + loss_b)

        elif algo == 'qrdqn':
            self.qrdqn_a._pending_reward = getattr(self.qrdqn_a, '_pending_reward', 0) + reward_a
            self.qrdqn_b._pending_reward = getattr(self.qrdqn_b, '_pending_reward', 0) + reward_b
            if is_decision_tick:
                ns_a = self.env.get_state('A')
                ns_b = self.env.get_state('B')
                self.qrdqn_a.store_reward(self.qrdqn_a._pending_reward, ns_a)
                self.qrdqn_b.store_reward(self.qrdqn_b._pending_reward, ns_b)
                self.qrdqn_a._pending_reward = 0.0
                self.qrdqn_b._pending_reward = 0.0
            if self.train_ppo and tick % 16 == 0 and tick > 0:
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

        algo = self.algorithm
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
            self.cmaes_a.save('models/cmaes_bank_a')
            self.cmaes_b.save('models/cmaes_bank_b')
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
            if self.env.tick % self._action_repeat == 0:
                state_a = self.env.get_state('A')
                state_b = self.env.get_state('B')
                actions_a = self.ppo_agent_a.select_action(state_a, self.env.bank_a)
                actions_b = self.ppo_agent_b.select_action(state_b, self.env.bank_b)
            else:
                actions_a = self._last_actions_a
                actions_b = self._last_actions_b
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
            if self.env.tick % self._action_repeat == 0:
                state_a = self.env.get_state('A')
                state_b = self.env.get_state('B')
                actions_a = self.dqn_a.select_action(state_a, self.env.bank_a)
                actions_b = self.dqn_b.select_action(state_b, self.env.bank_b)
            else:
                actions_a = self._last_actions_a
                actions_b = self._last_actions_b
        elif algo == 'reinforce':
            if self.env.tick % self._action_repeat == 0:
                state_a = self.env.get_state('A')
                state_b = self.env.get_state('B')
                actions_a = self.reinforce_a.select_action(state_a, self.env.bank_a)
                actions_b = self.reinforce_b.select_action(state_b, self.env.bank_b)
            else:
                actions_a = self._last_actions_a
                actions_b = self._last_actions_b
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
            if self.env.tick % self._action_repeat == 0:
                state_a = self.env.get_state('A')
                state_b = self.env.get_state('B')
                actions_a = self.asp_a.select_action(state_a, self.env.bank_a, self.env.tick)
                actions_b = self.asp_b.select_action(state_b, self.env.bank_b, self.env.tick)
            else:
                actions_a = self._last_actions_a
                actions_b = self._last_actions_b
        elif algo == 'qrdqn':
            if self.env.tick % self._action_repeat == 0:
                state_a = self.env.get_state('A')
                state_b = self.env.get_state('B')
                actions_a = self.qrdqn_a.select_action(state_a, self.env.bank_a)
                actions_b = self.qrdqn_b.select_action(state_b, self.env.bank_b)
            else:
                actions_a = self._last_actions_a
                actions_b = self._last_actions_b
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
        }

    def _auto_train_loop(self):
        cfg = self.auto_train_config
        for i in range(self.auto_train_total):
            if not self.auto_train_running:
                break
            self.auto_train_current = i + 1
            run_name = f"{cfg.get('base_name', 'AutoTrain')} #{self.auto_train_current}"
            self.start(
                name=run_name,
                algorithm=cfg.get('algorithm', 'ppo'),
                duration_minutes=cfg.get('duration', 60),
                start_hour=cfg.get('start_hour', 8.0),
                speed=cfg.get('speed', 5000),
                train_ppo=True,
                _reuse_agents=(i > 0),
            )
            time.sleep(0.2)
            while self.running and self.auto_train_running:
                time.sleep(0.5)
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
        info['sim_time_seconds'] = self.sim_start_time + (self.env.tick if self.env else 0)
        info['active_passengers'] = len(self.env.active_passengers) if self.env else 0
        return info
