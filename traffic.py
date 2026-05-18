import random
import math
from typing import List, Tuple
from config import (
    NUM_FLOORS, FLOOR_INDEX, FLOOR_NAMES,
    BANK_A_FLOOR_INDICES, BANK_B_FLOOR_INDICES,
    MAX_POPULATION
)


class TrafficProfile:


    OFFICE_FLOORS_A = [FLOOR_INDEX[str(f)] for f in range(8, 14)]
    OFFICE_FLOORS_B = [FLOOR_INDEX[str(f)] for f in range(2, 8)]
    OFFICE_FLOORS = OFFICE_FLOORS_A + OFFICE_FLOORS_B

    GROUND_FLOORS = [FLOOR_INDEX[f] for f in ['G', 'LG']]
    PARKING_FLOORS = [FLOOR_INDEX[f] for f in ['P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7']]
    ENTRY_FLOORS = GROUND_FLOORS + PARKING_FLOORS
    LUNCH_FLOORS = [FLOOR_INDEX['1'], FLOOR_INDEX['14']]

    @staticmethod
    def get_arrival_rate(sim_time_seconds: int) -> float:
        hour = sim_time_seconds / 3600.0


        rate = 0.05


        rate += 1.35 * math.exp(-0.5 * ((hour - 9.25) / 0.3) ** 2)


        rate += 0.65 * math.exp(-0.5 * ((hour - 12.75) / 0.35) ** 2)
        rate += 0.50 * math.exp(-0.5 * ((hour - 13.5) / 0.35) ** 2)


        rate += 1.15 * math.exp(-0.5 * ((hour - 17.75) / 0.35) ** 2)


        rate += 0.12 * math.exp(-0.5 * ((hour - 11.0) / 0.5) ** 2)


        rate += 0.08 * math.exp(-0.5 * ((hour - 15.5) / 0.5) ** 2)

        return max(rate, 0.02)

    @staticmethod
    def generate_trip(sim_time_seconds: int) -> Tuple[int, int]:
        hour = sim_time_seconds / 3600.0


        if 8.5 <= hour <= 10.0:
            origin = random.choice(TrafficProfile.ENTRY_FLOORS)
            dest = random.choice(TrafficProfile.OFFICE_FLOORS)
            return origin, dest


        if 12.0 <= hour <= 13.0:
            if random.random() < 0.7:
                origin = random.choice(TrafficProfile.OFFICE_FLOORS)
                dest = random.choice(TrafficProfile.LUNCH_FLOORS)
            else:
                origin = random.choice(TrafficProfile.LUNCH_FLOORS)
                dest = random.choice(TrafficProfile.OFFICE_FLOORS)
            return origin, dest


        if 13.0 <= hour <= 14.0:
            if random.random() < 0.7:
                origin = random.choice(TrafficProfile.LUNCH_FLOORS)
                dest = random.choice(TrafficProfile.OFFICE_FLOORS)
            else:
                origin = random.choice(TrafficProfile.OFFICE_FLOORS)
                dest = random.choice(TrafficProfile.LUNCH_FLOORS)
            return origin, dest


        if 17.0 <= hour <= 19.0:
            origin = random.choice(TrafficProfile.OFFICE_FLOORS)
            dest = random.choice(TrafficProfile.ENTRY_FLOORS)
            return origin, dest


        origin = random.choice(list(range(NUM_FLOORS)))
        dest = random.choice(list(range(NUM_FLOORS)))
        while dest == origin:
            dest = random.choice(list(range(NUM_FLOORS)))
        return origin, dest


class TrafficGenerator:

    def __init__(self, speed_multiplier: float = 1.0, custom_rate_override: float = None):
        self.speed_multiplier = speed_multiplier
        self.custom_rate_override = custom_rate_override
        self.fractional_arrivals = 0.0

    def generate_arrivals(self, sim_time: int) -> List[Tuple[int, int]]:
        if self.custom_rate_override is not None:
            rate = self.custom_rate_override
        else:
            rate = TrafficProfile.get_arrival_rate(sim_time)


        self.fractional_arrivals += rate
        n_arrivals = int(self.fractional_arrivals)
        self.fractional_arrivals -= n_arrivals

        trips = []
        for _ in range(n_arrivals):
            origin, dest = TrafficProfile.generate_trip(sim_time)
            if origin != dest:
                trips.append((origin, dest))

        return trips

    def generate_burst(self, floor: int, count: int, direction: str = 'random') -> List[Tuple[int, int]]:
        trips = []
        for _ in range(count):
            if direction == 'up':
                dest = random.choice([f for f in range(floor + 1, NUM_FLOORS)])
            elif direction == 'down':
                dest = random.choice([f for f in range(0, floor)])
            else:
                dest = random.choice([f for f in range(NUM_FLOORS) if f != floor])
            trips.append((floor, dest))
        return trips
