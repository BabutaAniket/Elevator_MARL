from simulation import SimulationRunner
import time

algos = ['scan', 'nearest', 'astar', 'astar_scan', 'ppo', 'dqn', 'reinforce',
         'qlearning', 'cmaes', 'roundrobin', 'idlewait', 'astar_scan_ppo', 'qrdqn']

for algo in algos:
    r = SimulationRunner()
    rid = r.start("test-" + algo, algo, duration_minutes=1, start_hour=9.0, speed=1000)
    time.sleep(3)
    r.stop()
    st = r.get_live_state()
    tick = st.get("tick", 0)
    waiting = st.get("total_waiting", 0)
    delivered = st.get("total_delivered", 0)
    avg_wait = st.get("avg_wait", 0)
    print(f"{algo:12s}: tick={tick:4d} wait={waiting:3d} delivered={delivered:3d} avg_wait={avg_wait:.1f}s")

print("\nAll algorithms OK")
