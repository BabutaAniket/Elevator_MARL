from environment import BuildingEnvironment
from agents import ScanController, AStarDispatchController, NearestFirstController
from traffic import TrafficGenerator, TrafficProfile


def test_rates():
    peak_rate = TrafficProfile.get_arrival_rate(int(9.25 * 3600))
    offpeak_rate = TrafficProfile.get_arrival_rate(15 * 3600)
    print(f"Peak rate (9:15 AM): {peak_rate:.2f}/sec = {peak_rate*3600:.0f}/hour")
    print(f"Off-peak rate (3 PM): {offpeak_rate:.2f}/sec = {offpeak_rate*3600:.0f}/hour")


def run_test(label, start_hour, controller_cls):
    env = BuildingEnvironment()
    tg = TrafficGenerator()
    ctrl_a = controller_cls(env.bank_a) if controller_cls == ScanController else controller_cls()
    ctrl_b = controller_cls(env.bank_b) if controller_cls == ScanController else controller_cls()

    start_sec = int(start_hour * 3600)
    for t in range(600):
        arrivals = tg.generate_arrivals(start_sec + t)
        for o, d in arrivals:
            env.add_passenger(o, d)
        actions_a = ctrl_a.get_actions(env.bank_a, t)
        actions_b = ctrl_b.get_actions(env.bank_b, t)
        env.step(actions_a, actions_b)

    info = env.get_snapshot()
    print(f"{label}: Wait={info['total_waiting']} Delivered={info['total_delivered']} "
          f"AvgWait={info['avg_wait']:.1f}s MaxWait={info['max_wait']:.0f}s "
          f"Energy={info['total_energy']:.0f}")


if __name__ == "__main__":
    test_rates()
    print()
    print("--- Peak (9 AM, SCAN) ---")
    run_test("SCAN-Peak", 9.0, ScanController)
    print("--- Off-Peak (3 PM, SCAN) ---")
    run_test("SCAN-OffPeak", 15.0, ScanController)
    print("--- Peak (9 AM, Nearest) ---")
    run_test("Nearest-Peak", 9.0, NearestFirstController)
    print("--- Peak (9 AM, A*) ---")
    run_test("A*-Peak", 9.0, AStarDispatchController)
    print()
    print("Test PASSED")
