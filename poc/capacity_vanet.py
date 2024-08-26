import itertools
import logging
import math
import time
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from dataclasses import dataclass
from functools import lru_cache
from multiprocessing import Pool
from typing import Optional, List, Dict

import mesa
import numpy as np
from matplotlib import pyplot as plt, patches
from matplotlib.patches import Rectangle, Circle
from mesa import Agent, Model
from mesa.space import ContinuousSpace

import poc.units as units
import poc.VanetTraceLoader as vanetLoader
import poc.simple as simple
from poc.VanetTraceLoader import VehicleTrace
from poc.scheduler import RandomActivationBySortedType

SEED = 42

# Constants for the simulation, adjusted for demonstration
TIME_STEP_MS = 1000  # Time step in milliseconds
TIME_STEP_S = TIME_STEP_MS / 1000.0  # Time step in seconds
STEPS_PER_SECOND = int(1 // TIME_STEP_S)  # Number of steps per second
assert np.isclose(STEPS_PER_SECOND * TIME_STEP_S, 1.0, rtol=1e-09, atol=1e-09), "Time step conversion error"
VEHICLE_SPEED_FAST_MS = 60 * (1000 / 3600)  # 60 km/h in m/s

VEC_STATION_COLORS = simple.VEC_STATION_COLORS


@dataclass
class RsuConfig:
    pos: tuple[int, int]
    range: int
    capacity: int


RSU_RANGE = 70
RSU_CAPACITY_T4 = 65 * units.TERA
RSU_CAPACITY_T4_HALF = 32.5 * units.TERA
RSU_CAPACITY_T4_QUARTER = 16.25 * units.TERA

CRETEIL_4_RSU_FULL_CAPA_CONFIG = [
    RsuConfig((35, 120), RSU_RANGE, RSU_CAPACITY_T4),  # blue
    RsuConfig((116, 150), RSU_RANGE, RSU_CAPACITY_T4),  # yellow
    RsuConfig((165, 50), RSU_RANGE, RSU_CAPACITY_T4),  # green
    RsuConfig((72, 50), RSU_RANGE, RSU_CAPACITY_T4),  # red
]

CRETEIL_4_RSU_HALF_CAPA_CONFIG = [
    RsuConfig((35, 120), RSU_RANGE, RSU_CAPACITY_T4_HALF),  # blue
    RsuConfig((116, 150), RSU_RANGE, RSU_CAPACITY_T4_HALF),  # yellow
    RsuConfig((165, 50), RSU_RANGE, RSU_CAPACITY_T4_HALF),  # green
    RsuConfig((72, 50), RSU_RANGE, RSU_CAPACITY_T4_HALF),  # red
]

CRETEIL_9_RSU_POSITIONS = [
    (35, 120),  # blue
    (116, 150),  # yellow
    (165, 50),  # green
    (72, 50),  # red
    (97, 98),  # olive
    (120, 30),  # yellow
    (30, 70),  # purple
    (70, 155),  # brown
    (160, 130),  # cyan
]

CRETEIL_9_RSU_FULL_CAPA_CONFIG = [RsuConfig(pos, RSU_RANGE, RSU_CAPACITY_T4) for pos in CRETEIL_9_RSU_POSITIONS]
CRETEIL_9_RSU_HALF_CAPA_CONFIG = [RsuConfig(pos, RSU_RANGE, RSU_CAPACITY_T4_HALF) for pos in CRETEIL_9_RSU_POSITIONS]
CRETEIL_9_RSU_QUARTER_CAPA_CONFIG = [RsuConfig(pos, RSU_RANGE, RSU_CAPACITY_T4_QUARTER) for pos in
                                     CRETEIL_9_RSU_POSITIONS]

CRETEIL_3_FAIL_RSU_FULL_CAPA_CONFIG = [
    RsuConfig((35, 120), RSU_RANGE, RSU_CAPACITY_T4),  # blue
    RsuConfig((116, 150), RSU_RANGE, RSU_CAPACITY_T4),  # yellow
    RsuConfig((165, 50), RSU_RANGE, RSU_CAPACITY_T4),  # green
]

CRETEIL_3_FAIL_RSU_HALF_CAPA_CONFIG = [
    RsuConfig((35, 120), RSU_RANGE, RSU_CAPACITY_T4_HALF),  # blue
    RsuConfig((116, 150), RSU_RANGE, RSU_CAPACITY_T4_HALF),  # yellow
    RsuConfig((165, 50), RSU_RANGE, RSU_CAPACITY_T4_HALF),  # green
]


class RSAgentStrategy(ABC):
    @abstractmethod
    def handle_offloading(self, station: "VECStationAgent"):
        pass

    def after_step(self, model: "VECModel"):
        # Default implementation does nothing.
        # Subclasses can override this method if needed.
        pass


class VehicleLoadGenerator(ABC):
    @abstractmethod
    def compute_offloaded_load(self, vehicle: "VehicleAgent"):
        """
        Compute the offloaded load of a vehicle.
        """
        pass


class StaticVehicleLoadGenerator(VehicleLoadGenerator):
    def compute_offloaded_load(self, vehicle: "VehicleAgent"):
        return 1


@lru_cache(maxsize=1)
def get_grid():
    return vanetLoader.get_grid()


def distance(pos1, pos2):
    """Calculate the Euclidean distance between two positions."""
    x1, y1 = pos1
    x2, y2 = pos2
    return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def is_moving_towards(vehicle_pos, vehicle_orientation, station_pos):
    # Convert vehicle orientation to radians for math functions
    rad = math.radians(vehicle_orientation)

    # Calculate direction vector of vehicle's movement
    vehicle_dir = (math.cos(rad), math.sin(rad))

    # Calculate vector from vehicle to station
    to_station_vec = (station_pos[0] - vehicle_pos[0], station_pos[1] - vehicle_pos[1])

    # Calculate dot product
    dot_product = vehicle_dir[0] * to_station_vec[0] + vehicle_dir[1] * to_station_vec[1]

    # Check if dot product is positive
    return dot_product > 0


class VehicleAgent(simple.VehicleAgent):
    """A vehicle agent that follows a list of waypoints and calculates its angle."""

    # TODO fix inheritance stuff
    def __init__(self, unique_id, model: "VECModel", trace: Optional[vanetLoader.VehicleTrace],
                 load_gen: VehicleLoadGenerator):
        super().__init__(unique_id, model, 0, [])
        self.offloaded_load = 0
        self.invocation = 0
        self.trace_i = 0
        self.trace = trace
        self.load_gen = load_gen
        self.station: Optional["VECStationAgent"] = None

        self.active = True
        if self.trace is None:
            self.active = False
            return

        # Necessary for determining initial station
        self.trace_iterator = iter(self.trace.trace.iterrows())
        first_trace = self.trace.trace.iloc[0]
        self.angle = first_trace['vehicle_angle']
        self.pos = (first_trace['vehicle_x'], first_trace['vehicle_y'])

    def do_step(self):
        _, state = next(self.trace_iterator)
        assert state['timestep_time'] == self.trace_i + self.trace.first_ts, "Time step mismatch"
        self.trace_i += 1

        self.pos = (state['vehicle_x'], state['vehicle_y'])
        self.angle = state['vehicle_angle']
        # print("Move to", self.pos, "with angle", self.angle)
        self.model.space.move_agent(self, self.pos)

        self.offloaded_load = self.load_gen.compute_offloaded_load(self)

    def step(self):
        # TODO check there are no "holes" in timestamp list for all vehicles
        if self.invocation % STEPS_PER_SECOND == 0:
            self.do_step()

        self.invocation += 1

        # external_ts = int(self.ts * TIME_STEP_S)
        # if self.trace.first_ts <= external_ts <= self.trace.last_ts:
        #     self.do_step(external_ts)
        # if external_ts > self.trace.last_ts:
        #     self.active = False

    def count_nearby_vehicles(self) -> int:
        count = 0
        nearby_dist = 5
        for agent in self.model.schedule.get_agents_by_type(VehicleAgent):
            if agent != self and distance(agent.pos, self.pos) <= nearby_dist:
                count += 1

        return count

    @property
    def rsu_distance(self):
        return distance(self.pos, self.station.pos)

    def __repr__(self):
        return f"VehicleAgent{self.unique_id}"


def calculate_trajectory_suitability(station: "VECStationAgent", vehicle: VehicleAgent):
    """
    Calculate the suitability of a vehicle's trajectory to the station.
    The suitability is a value between 0 and 1, where 1 is the best suitability.
    Returns 0 if the vehicle is not in range.
    """

    if not station.is_vehicle_in_range(vehicle):
        return 0

    bearing_rad = station.calculate_vehicle_station_bearing(vehicle)
    vehicle_angle_rad = math.radians(vehicle.angle)

    angle_at_vehicle = bearing_rad - vehicle_angle_rad

    result = 1 - (0.5 * math.cos(angle_at_vehicle) + 0.75) / 1.25 * distance(station.pos, vehicle.pos) / station.range

    assert 0 <= result <= 1, f"Handover metric is out of bounds: {result}"
    return result


def calculate_station_suitability(station: "VECStationAgent", station_load: float, vehicle: VehicleAgent):
    """
    Calculate the suitability of a station to receive a vehicle.
    The suitability is a value between 0 and 1, where 1 is the best suitability.
    Returns 0 if the station is not in range or the target station's capacity would be exceeded after HO.
    """

    assert station_load >= 0, f"Invalid negative station load {station_load}"
    if station_load + vehicle.offloaded_load > station.capacity:
        return 0

    capacity_suitability = max(0, (station.capacity - station_load - vehicle.offloaded_load) / station.capacity)
    trajectory_suitability = calculate_trajectory_suitability(station, vehicle)
    suitability = capacity_suitability * trajectory_suitability

    assert 0 <= suitability <= 1, f"Suitability score is out of bounds: {suitability}"
    return suitability


class VECStationAgent(simple.VECStationAgent):
    """A VEC station agent with a communication range."""

    # TODO fix inheritance stuff
    def __init__(self, unique_id, model: "VECModel", strategy: RSAgentStrategy, position, range_m, capacity,
                 neighbors=None, can_read_neighbor_load=False):
        super().__init__(unique_id, model, 0, 0, 0)
        self.strategy = strategy
        self.pos = position
        self.range = range_m
        self.capacity = capacity
        self.neighbors: List[VECStationAgent] = neighbors if neighbors else []
        self.vehicles: List[VehicleAgent] = []
        self.distance_threshold = 0.7  # Can be removed (except rendering)
        self.load_threshold = 0.95
        self.vehicle_distance = None
        self.neighbor_load = {x.unique_id: 0 for x in self.neighbors}
        self.can_read_neighbor_load = can_read_neighbor_load

    @property
    def load(self):
        return sum([vehicle.offloaded_load for vehicle in self.vehicles])

    @property
    def utilization(self):
        return self.load / self.capacity

    @property
    def connected_vehicles(self):
        return len(self.vehicles)

    def get_neighbor_load(self, neighbor_id):
        if self.can_read_neighbor_load:
            return [x.load for x in self.neighbors if x.unique_id == neighbor_id][0]
        return self.neighbor_load[neighbor_id]

    def increment_neighbor_load(self, neighbor_id, load):
        self.neighbor_load[neighbor_id] += load
        # Might be less than 0 because of delayed load sharing
        if self.neighbor_load[neighbor_id] < 0:
            self.neighbor_load[neighbor_id] = 0

    def step(self):
        self.strategy.handle_offloading(self)

    def calculate_vehicle_station_bearing(self, vehicle: VehicleAgent):
        # Calculate the difference in x and y coordinates
        dx = vehicle.pos[0] - self.pos[0]
        dy = vehicle.pos[1] - self.pos[1]

        # Calculate the angle in radians
        return math.atan2(dy, dx)

    def perform_handover(self, to: "VECStationAgent", vehicle: VehicleAgent, cause="range"):
        assert self != to, "Cannot hand over to the same station"
        self.vehicles.remove(vehicle)
        to.vehicles.append(vehicle)
        vehicle.station = to
        self.increment_neighbor_load(to.unique_id, vehicle.offloaded_load)
        to.increment_neighbor_load(self.unique_id, -vehicle.offloaded_load)

        # Stats
        # noinspection PyTypeChecker
        model: VECModel = self.model
        model.report_successful_handovers += 1
        model.report_total_successful_handovers += 1

        if cause == "range":
            model.report_total_successful_handovers_range += 1
        elif cause == "load_balancing":
            model.report_total_successful_handovers_load_balancing += 1
        elif cause == "overload":
            model.report_total_successful_handovers_overload += 1
        else:
            raise ValueError(f"Invalid cause for handover: {cause}")

    def report_failed_handover(self):
        # noinspection PyTypeChecker
        model: VECModel = self.model
        model.report_failed_handovers += 1
        model.report_total_failed_handovers += 1

    def request_handover(self, vehicle: VehicleAgent, force=False) -> bool:
        """
        A station requests that a vehicle be handed over to a neighbor station.
        The target station can accept or reject the request based on its own criteria.
        """

        # TODO for now handover can be forced
        if not force:
            # Check if the vehicle is 1. in range and 2. the RSU has enough capacity
            if distance(self.pos, vehicle.pos) > self.range:
                return False
            if self.load + vehicle.offloaded_load > self.capacity:
                return False

            # TODO: Implement more sophisticated check

        return True

    def __repr__(self):
        return f"VECStation{self.unique_id}"

    def is_vehicle_in_range(self, vehicle: VehicleAgent):
        return distance(self.pos, vehicle.pos) <= self.range


class VECModel(Model):
    """A model with a single vehicle following waypoints on a rectangular road layout."""

    def __init__(self, rs_strategy: RSAgentStrategy, rsu_configs: List[RsuConfig],
                 vehicle_load_gen: VehicleLoadGenerator, traces: Dict[str, VehicleTrace], load_update_interval=1,
                 start_at=0, **kwargs):
        # Seed is set via super().new()
        super().__init__()
        self.running = True
        self.rs_strategy = rs_strategy
        self.load_update_interval = load_update_interval
        self.traces = traces
        self.vehicle_load_gen = vehicle_load_gen
        self.width, self.height = vanetLoader.get_size()
        assert self.width == 200
        assert self.height == 200

        self.space = ContinuousSpace(self.width, self.height, False)  # Non-toroidal space
        self.schedule = RandomActivationBySortedType(self, [VehicleAgent, VECStationAgent])

        self.report_successful_handovers = 0
        self.report_failed_handovers = 0
        self.report_total_successful_handovers = 0
        self.report_total_successful_handovers_range = 0
        self.report_total_successful_handovers_load_balancing = 0
        self.report_total_successful_handovers_overload = 0
        self.report_total_failed_handovers = 0

        def station_vehicle_count_collector(a: Agent):
            if isinstance(a, VECStationAgent):
                return a.connected_vehicles
            return None

        def station_load_collector(a: Agent):
            if isinstance(a, VECStationAgent):
                return a.load / a.capacity
            return None

        def vehicle_load_collector(a: Agent):
            if isinstance(a, VehicleAgent):
                return a.offloaded_load
            return None

        self.datacollector = mesa.DataCollector(
            model_reporters={
                "SuccessfulHandoverCount": "report_successful_handovers",
                "TotalSuccessfulHandoverCount": "report_total_successful_handovers",
                "TotalSuccessfulHandoverCountRange": "report_total_successful_handovers_range",
                "TotalSuccessfulHandoverCountLoadBalancing": "report_total_successful_handovers_load_balancing",
                "TotalSuccessfulHandoverCountOverload": "report_total_successful_handovers_overload",
                "FailedHandoverCount": "report_failed_handovers",
                "TotalFailedHandoverCount": "report_total_failed_handovers",
                "AvgQoS": VECModel.report_avg_qos,
                "MinQoS": VECModel.report_min_qos,
                "AvgQoS_LoadBased": VECModel.report_avg_qos_load_based,
                "MinQoS_LoadBased": VECModel.report_min_qos_load_based,
                "AvgQoS_RangeBased": VECModel.report_avg_qos_range_based,
                "MinQoS_RangeBased": VECModel.report_min_qos_range_based,
                "GiniLoad": VECModel.report_gini_load,
                "VehicleCount": VECModel.report_vehicle_count,
            },
            agent_reporters={"Distances": "vehicle_distance", "StationVehicleCount": station_vehicle_count_collector,
                             "StationVehicleLoad": station_load_collector, "VehicleLoad": vehicle_load_collector}
        )

        self.vec_stations = []
        for i, rsu_config in enumerate(rsu_configs, start=1):
            station = VECStationAgent(10000 + i, self, self.rs_strategy, rsu_config.pos, rsu_config.range,
                                      rsu_config.capacity, can_read_neighbor_load=self.load_update_interval == 0)
            self.vec_stations.append(station)
            self.schedule.add(station)

        for i in range(len(self.vec_stations)):
            self.vec_stations[i].neighbors = [s for s in self.vec_stations if s != self.vec_stations[i]]
            self.vec_stations[i].neighbor_load = {x.unique_id: 0 for x in self.vec_stations[i].neighbors}

        self.vehicle_id = 0
        self.shared_load_info = {s.unique_id: 0 for s in self.vec_stations}

        self.unplaced_vehicles: List[vanetLoader.VehicleTrace] = [v for k, v in self.traces.items()]
        self.unplaced_vehicles.sort(key=lambda x: x.first_ts, reverse=True)

        self.to_remove: List[VehicleAgent] = []

        self.step_second = -1

        self.datacollector.collect(self)

        for _ in range(start_at):
            self.step()

    def spawn_vehicle(self, trace_id, step) -> Optional[VehicleAgent]:
        self.vehicle_id += 1
        vehicle = VehicleAgent(self.vehicle_id, self, self.traces[trace_id], self.vehicle_load_gen)
        vehicle.ts = step // TIME_STEP_S

        self.schedule.add(vehicle)

        station = min(self.vec_stations, key=lambda x: distance(x.pos, vehicle.pos))
        station.vehicles.append(vehicle)
        vehicle.station = station

        return vehicle

    def step(self):

        while self.to_remove and self.to_remove[-1].trace.last_ts == self.step_second:
            v = self.to_remove.pop()
            v.station.vehicles.remove(v)
            self.schedule.remove(v)
            v.remove()

        assert len(self.to_remove) == len(self.agents) - len(
            self.vec_stations), "Agent count mismatch"  # 4 is number of stations

        self.step_second += 1

        while self.unplaced_vehicles and self.unplaced_vehicles[-1].first_ts == self.step_second:
            v_trace = self.unplaced_vehicles.pop()
            v = self.spawn_vehicle(v_trace.id, self.step_second)
            if not v:
                continue

            # Insert while sorting on last_ts
            self.to_remove.append(v)
            self.to_remove.sort(key=lambda x: x.trace.last_ts, reverse=True)

        if self.load_update_interval > 0 and self.step_second % self.load_update_interval == 0:
            self.update_shared_load_info()

        # TODO simplify??
        for _ in range(STEPS_PER_SECOND):
            self.schedule.step()
            self.rs_strategy.after_step(self)

        self.datacollector.collect(self)

        # Reset per-step statistics
        self.report_successful_handovers = 0
        self.report_failed_handovers = 0

        # Terminate if no more vehicles are left
        if len(self.to_remove) == 0 and len(self.unplaced_vehicles) == 0:
            self.running = False

    def update_shared_load_info(self):
        for station in self.vec_stations:
            for neighbor in self.vec_stations:
                if neighbor in station.neighbor_load or neighbor in station.neighbors:
                    station.neighbor_load[neighbor.unique_id] = neighbor.load

    def report_avg_qos(self):
        qos = compute_qos(self)
        if len(qos) == 0:
            return 1
        result = sum(qos) / len(qos)
        other_qos = [x * y for x, y in zip(compute_load_based_qos(self), compute_range_based_qos(self))]
        other_result = sum(other_qos) / len(other_qos)
        assert abs(result - other_result) < 1e-9, f"QoS mismatch: {result} vs {other_result}"
        return sum(qos) / len(qos)

    def report_min_qos(self):
        qos = compute_qos(self)
        return min(qos, default=1)

    def report_avg_qos_load_based(self):
        qos = compute_load_based_qos(self)
        if len(qos) == 0:
            return 1
        return sum(qos) / len(qos)

    def report_min_qos_load_based(self):
        qos = compute_load_based_qos(self)
        return min(qos, default=1)

    def report_avg_qos_range_based(self):
        qos = compute_range_based_qos(self)
        if len(qos) == 0:
            return 1
        return sum(qos) / len(qos)

    def report_min_qos_range_based(self):
        qos = compute_range_based_qos(self)
        return min(qos, default=1)

    def report_gini_load(self):
        loads = [station.load for station in self.vec_stations]
        return compute_gini(loads)

    def report_vehicle_count(self):
        return len(self.agents) - len(self.vec_stations)


def compute_gini(array):
    """Compute Gini coefficient of an array."""
    array = np.array(array, dtype=np.float64)
    if np.amin(array) < 0:
        array -= np.amin(array)  # Values cannot be negative
    array += 0.0000001  # Values cannot be 0
    array = np.sort(array)  # Sort smallest to largest
    index = np.arange(1, array.shape[0] + 1)  # Index per array element
    n = array.shape[0]
    return (np.sum((2 * index - n - 1) * array)) / (n * np.sum(array))


def compute_load_based_qos(model: VECModel) -> List[float]:
    """
    Load-based QoS is 1 if the station is not overloaded, otherwise the capacity divided by the load.
    """

    qos_list = []
    for agent in model.schedule.get_agents_by_type(VehicleAgent):
        qos = 1
        if agent.station.load > agent.station.capacity:
            qos = agent.station.capacity / agent.station.load
        qos_list.append(qos)

    return qos_list


def compute_range_based_qos(model: VECModel) -> List[float]:
    """
    Range-based QoS is 1 if the vehicle is within the station's range, otherwise a decaying function of the distance.
    """

    alpha = 0.0231

    qos_list = []
    for agent in model.schedule.get_agents_by_type(VehicleAgent):
        qos = 1
        if not agent.station.is_vehicle_in_range(agent):
            qos = math.exp(-alpha * (agent.rsu_distance - agent.station.range))

        qos_list.append(qos)

    return qos_list


RANGE_QOS_ALPHA = 0.0231


def compute_vehicle_qos(agent: VehicleAgent):
    """
    Compute QoS as the load of the station divided by its capacity if the vehicle is within range, otherwise 0.
    """

    range_factor = 1
    dist = distance(agent.pos, agent.station.pos)
    if not agent.station.is_vehicle_in_range(agent):
        range_factor = math.exp(-RANGE_QOS_ALPHA * (dist - agent.station.range))

    load_factor = 1
    if agent.station.load > agent.station.capacity:
        load_factor = agent.station.capacity / agent.station.load

    return load_factor * range_factor


def compute_qos(model: VECModel) -> List[float]:
    """
    Compute QoS as the load of the station divided by its capacity if the vehicle is within range, otherwise 0.
    """

    qos_list = []
    for agent in model.schedule.get_agents_by_type(VehicleAgent):
        qos = compute_vehicle_qos(agent)
        qos_list.append(qos)

    return qos_list


class DefaultOffloadingStrategy(RSAgentStrategy):
    def __init__(self, overload_threshold=0.95, leaving_threshold=0.05, imp_ho_timer=10, alt_ho_hysteresis=0.1,
                 alt_suitability_min=0.2):
        self.overload_threshold = overload_threshold
        self.leaving_threshold = leaving_threshold
        # self.imp_ho_timer = imp_ho_timer
        # TODO rename to util_hysteresis
        self.alt_ho_hysteresis = alt_ho_hysteresis
        self.alt_suitability_min = alt_suitability_min
        # self.next_ho_timer = defaultdict(int)

    def handle_offloading(self, station: VECStationAgent):

        # Step 0: Decrement HO timers
        # for vehicle in station.vehicles:
        #     # TODO remove HO timer
        #     self.next_ho_timer[vehicle.unique_id] -= 1

        # Step 1: Hand-over vehicles that are leaving anyway
        self.handle_vehicle_leaving_range(station)

        # Step 2: Hand-over vehicles to neighboring stations considering load balancing (also handles overload)
        self.handle_load_balancing_with_neighbors(station)

        # Step 3: Manage overload by prioritizing vehicles for handover
        # if station.load > station.load_threshold * station.capacity:
        #     self.manage_overload(station)

    def handle_vehicle_leaving_range(self, station):
        """
        Handle vehicles that are leaving the station's range.
        Iterate through vehicles and hand over if the trajectory suitability is below a certain threshold.
        If the suitability is below the leaving threshold, the handover is forced.
        """

        for vehicle in list(station.vehicles):
            # Check imperative HO timer
            # TODO investigate leaving_ho_timer
            # if self.next_ho_timer[vehicle.unique_id] > 0:
            #     continue

            # Based on trajectory suitability, decide if the vehicle should be handed over
            trajectory_suitability = calculate_trajectory_suitability(station, vehicle)
            if trajectory_suitability <= self.leaving_threshold:
                self.attempt_handover_vehicle(station, vehicle, "range", force=not station.is_vehicle_in_range(vehicle))

    def handle_load_balancing_with_neighbors(self, current: VECStationAgent):
        # already_gone = set()

        stations_with_vehicles = itertools.product(current.neighbors, current.vehicles)
        stations_vehicles_suitability = [
            (calculate_station_suitability(s, current.get_neighbor_load(s.unique_id), v), s, v)
            for s, v in stations_with_vehicles]
        stations_vehicles_suitability.sort(key=lambda x: x[0], reverse=True)

        for suitability, neighbor_station, vehicle in stations_vehicles_suitability:
            is_overload = False
            # if vehicle in already_gone:
            #     continue

            # Automatically skip if the vehicle is out of range of the neighbor
            if not neighbor_station.is_vehicle_in_range(vehicle):
                continue

            neighbor_utilization = (current.get_neighbor_load(neighbor_station.unique_id) +
                                    vehicle.offloaded_load) / neighbor_station.capacity

            # In case the station is not overloaded, we can quit load-balancing if the suitability (which is descending)
            # is less than the required minimum
            # We only consider handovers to be overload-related if they wouldn't already be performed by
            # the default load-balancing behavior
            if suitability < self.alt_suitability_min:
                if current.utilization < self.overload_threshold:
                    break
                is_overload = True

            # Quit if neighbor station has more load than current (considering hysteresis)
            if neighbor_utilization > current.utilization - self.alt_ho_hysteresis:
                break

            success = neighbor_station.request_handover(vehicle, force=is_overload)
            if not success:
                current.report_failed_handover()
                continue

            logging.info(
                f"Vehicle {vehicle.unique_id} is being handed over to VEC station {neighbor_station.unique_id} to balance load")
            current.perform_handover(neighbor_station, vehicle, "overload" if is_overload else "load_balancing")
            # self.next_ho_timer[vehicle.unique_id] = self.imp_ho_timer
            # already_gone.add(vehicle)

            # Recursive call to perform potentially multiple load-balancing related handovers
            self.handle_load_balancing_with_neighbors(current)
            break

    # def manage_overload(self, station: VECStationAgent):

    # Iterate through vehicles sorted by trajectory suitability ascending, selects the least suitable first
    # # TODO what if other station is also overloaded???
    # for vehicle in sorted(station.vehicles, key=lambda x: calculate_trajectory_suitability(station, x),
    #                       reverse=False):
    #     if self.next_ho_timer[vehicle.unique_id] > 0:
    #         continue
    #
    #     if station.load < station.load_threshold * station.capacity:
    #         return
    #
    #     logging.info(f"Vehicle {vehicle.unique_id} is being considered for handover due to overload")
    #
    #     self.attempt_handover_vehicle(station, vehicle, "overload", force=True)

    def attempt_handover_vehicle(self, station: VECStationAgent, vehicle: VehicleAgent, cause, force=False) -> bool:

        neighbors_with_score = [
            (x, calculate_station_suitability(x, station.get_neighbor_load(x.unique_id), vehicle))
            for x in station.neighbors if
            distance(x.pos, vehicle.pos) < x.range]
        neighbors_with_score.sort(key=lambda x: x[1], reverse=True)

        if len(neighbors_with_score) == 0:
            if force:
                logging.warning(f"Vehicle {vehicle.unique_id} is leaving coverage area!!")
            return False

        logging.debug(f"Neighbors with score for vehicle {vehicle.unique_id}: %s", neighbors_with_score)

        if neighbors_with_score[0][1] == 0 and not force:
            logging.warning(f"Vehicle {vehicle.unique_id} cannot be handed over to any neighbor (no force)")
            return False

        # Loop through sorted neighbors and handover to the first one that accepts
        for neighbor, score in neighbors_with_score:
            success = neighbor.request_handover(vehicle, force)
            if not success:
                station.report_failed_handover()
                continue

            station.perform_handover(neighbor, vehicle, cause)
            logging.info(f"Vehicle {vehicle.unique_id} handed over to VEC station {neighbor.unique_id}")
            # self.next_ho_timer[vehicle.unique_id] = self.imp_ho_timer

            return True

        return False


class NearestRSUStrategy(RSAgentStrategy):
    def handle_offloading(self, station: VECStationAgent):
        # A vehicle should always be connected to the nearest RSU
        # Check for all vehicles that the nearest RSU is the current, otherwise hand over to nearest
        for vehicle in list(station.vehicles):
            nearest_station = min(station.neighbors, key=lambda x: distance(x.pos, vehicle.pos))
            if distance(nearest_station.pos, vehicle.pos) < distance(station.pos, vehicle.pos):
                logging.info(
                    f"Vehicle {vehicle.unique_id} is being handed over to the nearest station {nearest_station.unique_id}")
                station.perform_handover(nearest_station, vehicle)

    def after_step(self, model: VECModel):
        # Assert that every vehicle is connected to the nearest station
        vehicles = model.schedule.get_agents_by_type(VehicleAgent)
        stations = model.vec_stations

        for vehicle in vehicles:
            nearest_station = min(stations, key=lambda x: distance(x.pos, vehicle.pos))
            assert distance(nearest_station.pos, vehicle.pos) == distance(vehicle.station.pos, vehicle.pos), \
                f"Vehicle {vehicle.unique_id} is not connected to the nearest station"


class LatestPossibleHandoverStrategy(RSAgentStrategy):
    def handle_offloading(self, station: VECStationAgent):
        # We know that all vehicles move before the RSU handover phase
        # Therefore, we only need to check if some vehicles are out of the range of the RSU
        # If so, perform handover to the nearest other RSU
        for vehicle in list(station.vehicles):
            if vehicle.rsu_distance > station.range:
                in_range_stations = [s for s in station.neighbors if s.is_vehicle_in_range(vehicle)]
                if not in_range_stations:
                    logging.warning(f"Vehicle {vehicle.unique_id} is out of range of all RSUs")
                    continue

                nearest_station = min(in_range_stations, key=lambda x: distance(x.pos, vehicle.pos))
                if nearest_station == station:
                    logging.warning(f"Vehicle {vehicle.unique_id} is out of range of all RSUs")
                    continue
                logging.info(
                    f"Vehicle {vehicle.unique_id} is being handed over to the nearest station {nearest_station.unique_id}")
                station.perform_handover(nearest_station, vehicle)

    def after_step(self, model: "VECModel"):
        # Check that each vehicle is in range of its station
        for vehicle in model.schedule.get_agents_by_type(VehicleAgent):
            assert (vehicle.station.is_vehicle_in_range(vehicle)
                    or not [s for s in model.vec_stations if s.is_vehicle_in_range(vehicle)]), \
                f"Vehicle {vehicle.unique_id} is out of range"


class EarliestPossibleHandoverStrategy(RSAgentStrategy):
    def __init__(self, max_recent_connections=2):
        # Track a deque of RSUs to which vehicles were recently connected
        self.previously_connected = defaultdict(lambda: deque(maxlen=max_recent_connections))
        self.max_recent_connections = max_recent_connections

    def handle_offloading(self, station: VECStationAgent):
        # For each vehicle, check if another RSU is in range which wasn't previously connected
        # If so, perform handover to closest
        for vehicle in list(station.vehicles):
            # Filter stations that are in range
            # todo check on moving towards
            in_range_stations = [x for x in station.neighbors
                                 if x.unique_id not in self.previously_connected[vehicle.unique_id]
                                 and distance(x.pos, vehicle.pos) <= x.range
                                 and is_moving_towards(vehicle.pos, vehicle.angle, x.pos)]

            if not in_range_stations:
                if station.is_vehicle_in_range(vehicle):
                    continue

                # TODO still somewhat bugged
                # Special case: All stations in range are in previous connections
                in_range_stations = [x for x in station.neighbors
                                     if distance(x.pos, vehicle.pos) <= x.range]

                if not in_range_stations:
                    logging.warning(f"Vehicle {vehicle.unique_id} is out of range of all RSUs")
                    continue

            # Get the closest station that wasn't previously connected
            nearest_station = min(in_range_stations, key=lambda x: distance(x.pos, vehicle.pos))

            # print(station.unique_id, "->", nearest_station.unique_id)
            logging.info(
                f"Vehicle {vehicle.unique_id} is being handed over to the nearest station {nearest_station.unique_id}")
            station.perform_handover(nearest_station, vehicle)
            self.previously_connected[vehicle.unique_id].append(station.unique_id)

    def after_step(self, model: "VECModel"):
        # Check that each vehicle is in range of its station
        for vehicle in model.schedule.get_agents_by_type(VehicleAgent):
            assert (vehicle.station.is_vehicle_in_range(vehicle)
                    or not [s for s in model.vec_stations if s.is_vehicle_in_range(vehicle)]), \
                f"Vehicle {vehicle.unique_id} is out of range"


class EarliestPossibleHandoverNoBackStrategy(RSAgentStrategy):
    def __init__(self, max_recent_connections=2):
        # Track a deque of RSUs to which vehicles were recently connected
        self.previously_connected = defaultdict(lambda: deque(maxlen=max_recent_connections))
        self.max_recent_connections = max_recent_connections

    def handle_offloading(self, station: VECStationAgent):
        # For each vehicle, check if another RSU is in range which wasn't previously connected
        # If so, perform handover to closest
        for vehicle in list(station.vehicles):
            # Filter stations that are in range
            # todo check on moving towards
            in_range_stations = [x for x in station.neighbors
                                 if x.unique_id not in self.previously_connected[vehicle.unique_id]
                                 and distance(x.pos, vehicle.pos) <= x.range]

            if not in_range_stations:
                continue

            # Get the closest station that wasn't previously connected
            nearest_station = min(in_range_stations, key=lambda x: distance(x.pos, vehicle.pos))

            # print(station.unique_id, "->", nearest_station.unique_id)
            logging.info(
                f"Vehicle {vehicle.unique_id} is being handed over to the nearest station {nearest_station.unique_id}")
            station.perform_handover(nearest_station, vehicle)
            self.previously_connected[vehicle.unique_id].append(station.unique_id)


class DynamicVehicleLoadGenerator(VehicleLoadGenerator):
    local_computation = 0.91 * units.TERA
    load_min = 1.9 * units.TERA
    load_max = 3 * units.TERA

    def __init__(self, seed=42):
        self.load_per_vehicle = {}
        self.rng = np.random.default_rng(seed)

    def compute_offloaded_load(self, vehicle: "VehicleAgent"):
        if vehicle.unique_id not in self.load_per_vehicle:
            self.load_per_vehicle[vehicle.unique_id] = self.rng.uniform(self.load_min, self.load_max)

        return self.load_per_vehicle[vehicle.unique_id] - self.local_computation


def main():
    # Configuration for demonstration
    road_width = 200  # meters
    road_height = 200  # meters
    vehicle_speed = VEHICLE_SPEED_FAST_MS

    # Initialize and run the model
    model = VECModel(road_width, road_height, vehicle_speed, None)
    output = []

    vehicle_positions = []  # For recording vehicle positions
    vehicle_angles = []  # For recording vehicle angles

    # render_distance_chart(model)

    # Run the simulation for 200 steps to observe the vehicle's movement and rotation
    for i in range(1000):
        model.step()
    #     vehicle: VehicleAgent = model.vehicle
    #     if i % 2 == 0 and vehicle and vehicle.active:  # Collect position and angle for every 20 steps
    #         output.append(f"Step {i}: Vehicle Position: {vehicle.pos}, Angle: {vehicle.angle:.2f}")
    #         vehicle_positions.append(vehicle.pos)
    #         vehicle_angles.append(vehicle.angle)
    #
    # print(*output, sep='\n')

    # Visualize the recorded vehicle positions and directions
    fig, ax = plt.subplots()

    # Plot grid with cmap gray
    ax.imshow(get_grid(), cmap='gray')

    # road = patches.Polygon(model.waypoints, closed=True, fill=False, edgecolor='black', linewidth=2)
    # ax.add_patch(road)

    # Draw stations
    station_size = 6  # Size of the square

    for station in model.vec_stations:
        color = VEC_STATION_COLORS[station.unique_id]
        # Draw the station as a square
        lower_left_corner = (station.pos[0] - station_size / 2, station.pos[1] - station_size / 2)
        square = patches.Rectangle(lower_left_corner, station_size, station_size, color=color)
        ax.add_patch(square)

        # Draw the range as a dotted circle
        circle = plt.Circle(station.pos, station.range, color=color, fill=False, linestyle='dotted')
        ax.add_artist(circle)

    # Draw vehicles
    # vehicle_length = 2  # Length of the vehicle arrow

    # print("Vehicle positions")

    # for position, angle in zip(vehicle_positions, vehicle_angles):
    #     x, y = position
    #     dx = vehicle_length * np.cos(np.radians(angle))
    #     dy = vehicle_length * np.sin(np.radians(angle))
    #     ax.arrow(x, y, dx, dy, head_width=2, head_length=1, fc='blue', ec='blue', linewidth=2)
    #
    #     print((angle + 360) % 360)

    ax.set_xlim(0, road_width)
    ax.set_ylim(0, road_height)
    # ax.set_aspect('equal')
    plt.xlabel('x (meters)')
    plt.ylabel('y (meters)')
    plt.title('Vehicle Movement and Rotation')
    plt.grid(True)
    plt.show()


def extract_model_metrics(model, model_name):
    """
    Prints the evaluation metrics for a given model.

    Parameters:
    - model: The model object to extract metrics from.
    - model_name: A string representing the name or identifier of the model.
    """
    df = model.datacollector.get_model_vars_dataframe()
    print(f"{model_name} Success: {df['TotalSuccessfulHandoverCount'].iloc[-1]}")
    print(f"{model_name} Failed: {df['TotalFailedHandoverCount'].iloc[-1]}")
    print(f"{model_name} QoS: {df['AvgQoS'].mean()}")
    print(f"{model_name} QoSMin: {df['MinQoS'].mean()}")
    print(f"{model_name} Gini: {df['GiniLoad'].mean()}")

    return [
        model_name,
        df['TotalSuccessfulHandoverCount'].iloc[-1],
        df['TotalSuccessfulHandoverCountRange'].iloc[-1],
        df['TotalSuccessfulHandoverCountLoadBalancing'].iloc[-1],
        df['TotalSuccessfulHandoverCountOverload'].iloc[-1],
        df['TotalFailedHandoverCount'].iloc[-1],
        df['AvgQoS'].mean(),
        df['AvgQoS'].std(),
        df['MinQoS'].mean(),
        df['MinQoS'].std(),
        df['AvgQoS_LoadBased'].mean(),
        df['AvgQoS_RangeBased'].mean(),
        df['GiniLoad'].mean(),
        df['GiniLoad'].std()
    ]


STRATEGIES_DICT = {
    # "default": DefaultOffloadingStrategy,
    "default": DefaultOffloadingStrategy,
    "nearest": NearestRSUStrategy,
    "earliest": EarliestPossibleHandoverStrategy,
    "earliest2": EarliestPossibleHandoverNoBackStrategy,
    "latest": LatestPossibleHandoverStrategy,
}

SIMULATION_CONFIGS = {
    "creteil-morning": {
        "traces": lambda: vanetLoader.get_traces(morning=True, eval=True),
        "4-full": CRETEIL_4_RSU_FULL_CAPA_CONFIG,
        "4-half": CRETEIL_4_RSU_HALF_CAPA_CONFIG,
        "9-full": CRETEIL_9_RSU_FULL_CAPA_CONFIG,
        "9-half": CRETEIL_9_RSU_HALF_CAPA_CONFIG,
        "9-quarter": CRETEIL_9_RSU_QUARTER_CAPA_CONFIG,
        "3-fail-full": CRETEIL_3_FAIL_RSU_FULL_CAPA_CONFIG,
        "3-fail-half": CRETEIL_3_FAIL_RSU_HALF_CAPA_CONFIG,
    },
    "creteil-evening": {
        "traces": lambda: vanetLoader.get_traces(morning=False, eval=True),
        "4-full": CRETEIL_4_RSU_FULL_CAPA_CONFIG,
        "4-half": CRETEIL_4_RSU_HALF_CAPA_CONFIG,
        "9-full": CRETEIL_9_RSU_FULL_CAPA_CONFIG,
        "9-half": CRETEIL_9_RSU_HALF_CAPA_CONFIG,
        "9-quarter": CRETEIL_9_RSU_QUARTER_CAPA_CONFIG,
        "3-fail-full": CRETEIL_3_FAIL_RSU_FULL_CAPA_CONFIG,
        "3-fail-half": CRETEIL_3_FAIL_RSU_HALF_CAPA_CONFIG,
    }
}


def run_model(params, max_steps=None):
    logging.disable(logging.CRITICAL)

    scenario, rsu_config_name, model_name, strategy_key, load_update_interval, seed, _, strategy_config = params
    strategy_class = STRATEGIES_DICT[strategy_key]

    if strategy_key == 'default':
        strategy = strategy_class(**strategy_config)
    else:
        strategy = strategy_class()

    trace_loader = SIMULATION_CONFIGS[scenario]["traces"]
    rsu_config = SIMULATION_CONFIGS[scenario][rsu_config_name]

    model = VECModel(strategy, rsu_config, DynamicVehicleLoadGenerator(seed=seed), trace_loader(),
                     load_update_interval=load_update_interval, seed=seed)
    step = 0
    while model.running and (max_steps is None or step <= max_steps):
        model.step()
        step += 1

    filename = f"../results/runs/result_{scenario}_{rsu_config_name}_{model_name.lower()}"
    model.datacollector.get_model_vars_dataframe().to_csv(f"{filename}_model_vars.csv")
    # model.datacollector.get_agent_vars_dataframe().to_csv(f"{filename}_agent_vars.csv")

    return params, extract_model_metrics(model, model_name)


# Define parameter ranges for DefaultOffloadingStrategy
overload_threshold_values = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
leaving_threshold_values = [0]
imp_ho_timer_values = [0]
alt_ho_hysteresis_values = [0, 0.05, 0.1]
alt_suitability_min_values = [0.2, 0.25, 0.3, 0.35, 0.4]


def generate_default_strategy_configs(scenario, rsu_config):
    param_grid = itertools.product(
        overload_threshold_values,
        leaving_threshold_values,
        imp_ho_timer_values,
        alt_ho_hysteresis_values,
        alt_suitability_min_values
    )

    strategies = []
    for params in param_grid:
        config = {
            'overload_threshold': params[0],
            'leaving_threshold': params[1],
            'imp_ho_timer': params[2],
            'alt_ho_hysteresis': params[3],
            'alt_suitability_min': params[4]
        }
        name = f"Default_ovl{params[0]}_lvg{params[1]}_ho{params[2]}_hst{params[3]}_suit{params[4]}"
        strategies.append((scenario, rsu_config, name, 'default', 1, SEED, None, config))

    return strategies


BEST_DEFAULT_CONFIG = {
    'overload_threshold': 0.7,
    'leaving_threshold': 0,
    'alt_ho_hysteresis': 0.05,
    'alt_suitability_min': 0.3,
}


def store_results(results, filename):
    results.sort(key=lambda x: (not x[1][0].endswith('-Oracle'), x[1][0]))

    min_handovers = min(results, key=lambda x: x[1][1])[1][1]
    max_qos_mean = max(results, key=lambda x: x[1][6])[1][6]
    max_qos_min = max(results, key=lambda x: x[1][8])[1][8]
    min_gini = min(results, key=lambda x: x[1][12])[1][12]

    # Compute score of each entry by multiplying the % it achieves of the best score of each
    for _, result in results:
        # Compute the combined score for each entry; consider the difference between min and max values; should be <= 1
        ho_score = min_handovers / result[1]
        qos_mean_score = result[6] / max_qos_mean
        qos_min_score = result[8] / max_qos_min
        gini_score = min_gini / result[12]
        result.append(ho_score + qos_mean_score + qos_min_score + gini_score)
        result.append(ho_score * qos_mean_score * qos_min_score * gini_score)

    # Write to CSV with header line
    header = ("Model,HO_Total,HO_Range,HO_LB,HO_Overload,HO_Failed,AvgQoSMean,AvgQoSStd,MinQoSMean,MinQoSStd"
              ",AvgQoS_Load,AvgQoS_Range,GiniMean,GiniStd,EvalSum,EvalProd\n")
    header_len = len(header.split(","))
    with open(f"../results/{filename}.csv", "w") as f:
        f.write(header)
        for result in results:
            assert len(result[1]) == header_len, f"Length mismatch: {len(result[1])} vs {header_len}"
            f.write(",".join(map(str, result[1])) + "\n")


def create_run_model_with_steps(max_steps):
    def run_model_with_steps(params):
        return run_model(params, max_steps)

    return run_model_with_steps


def run_model_1000(params):
    return run_model(params, max_steps=1000)


def eval_strategy_params():
    start = time.time()

    scenario = "creteil-morning"
    rsu_config = "4-half"

    default_strategies = generate_default_strategy_configs(scenario, rsu_config)
    strategies = [
                     (scenario, rsu_config, "NearestRSU", "nearest", 1, SEED, 1388, None),
                     (scenario, rsu_config, "EarliestHO", "earliest", 1, SEED, 1540, None),
                     (scenario, rsu_config, "EarliestHONoBack", "earliest2", 1, SEED, 1494, None),
                     (scenario, rsu_config, "LatestHO", "latest", 1, SEED, 1264, None),
                 ] + default_strategies

    i = 0
    results = []
    if len(strategies) == 1:
        # Run in same thread for debugging
        results.append(run_model(strategies[0]))
    else:
        print("Start multi-threaded execution")
        with Pool(7) as p:
            # Run only 500 steps for param evaluation
            for res in p.imap_unordered(run_model, strategies):
                i += 1
                print(i, "/", len(strategies))
                results.append(res)

    print("Time elapsed:", int(time.time() - start), "s")

    store_results(results, f"results_eval_params_{scenario}_{rsu_config}")


def run_benchmarks(scenario, rsu_config):
    start = time.time()

    strategies = [
        (scenario, rsu_config, "ARHC-Oracle", "default", 0, SEED, None, BEST_DEFAULT_CONFIG),
        (scenario, rsu_config, "ARHC-01s", "default", 1, SEED, None, BEST_DEFAULT_CONFIG),
        (scenario, rsu_config, "ARHC-02s", "default", 2, SEED, None, BEST_DEFAULT_CONFIG),
        (scenario, rsu_config, "ARHC-03s", "default", 3, SEED, None, BEST_DEFAULT_CONFIG),
        (scenario, rsu_config, "ARHC-04s", "default", 4, SEED, None, BEST_DEFAULT_CONFIG),
        (scenario, rsu_config, "ARHC-05s", "default", 5, SEED, None, BEST_DEFAULT_CONFIG),
        (scenario, rsu_config, "ARHC-10s", "default", 10, SEED, None, BEST_DEFAULT_CONFIG),
        (scenario, rsu_config, "ARHC-15s", "default", 15, SEED, None, BEST_DEFAULT_CONFIG),
        (scenario, rsu_config, "ARHC-20s", "default", 20, SEED, None, BEST_DEFAULT_CONFIG),
        (scenario, rsu_config, "ARHC-25s", "default", 25, SEED, None, BEST_DEFAULT_CONFIG),
        (scenario, rsu_config, "ARHC-30s", "default", 30, SEED, None, BEST_DEFAULT_CONFIG),
        (scenario, rsu_config, "NearestRSU", "nearest", 1, SEED, 1388, None),
        (scenario, rsu_config, "EarliestHO", "earliest", 1, SEED, 1540, None),
        # (scenario, rsu_config, "EarliestHONoBack", "earliest2", 1, SEED, 1494, None),
        (scenario, rsu_config, "LatestHO", "latest", 1, SEED, 1264, None),
    ]

    i = 0
    results = []
    if len(strategies) == 1:
        # Run in same thread for debugging
        results.append(run_model(strategies[0]))
    else:
        print("Start multi-threaded execution")
        with Pool(7) as p:
            for res in p.imap_unordered(run_model, strategies):
                i += 1
                print(i, "/", len(strategies))
                results.append(res)

    print("Time elapsed:", int(time.time() - start), "s")

    store_results(results, f"results_{scenario}_{rsu_config}")


def run_all_benchmarks():
    configs = [
        ("creteil-morning", "4-full"),
        ("creteil-morning", "4-half"),
        ("creteil-evening", "4-full"),
        ("creteil-evening", "4-half"),
        ("creteil-morning", "9-full"),
        ("creteil-morning", "9-half"),
        ("creteil-morning", "9-quarter"),
        ("creteil-evening", "9-full"),
        ("creteil-evening", "9-half"),
        ("creteil-evening", "9-quarter"),
        ("creteil-morning", "3-fail-full"),
        ("creteil-morning", "3-fail-half"),
        ("creteil-evening", "3-fail-full"),
        ("creteil-evening", "3-fail-half"),
    ]

    for scenario, rsu_config in configs:
        print(f"Running benchmarks for {scenario} with {rsu_config}")
        run_benchmarks(scenario, rsu_config)


def investigate_min_qos(trace, rsu_config_name, strategy):
    trace_loader = SIMULATION_CONFIGS[trace]["traces"]
    rsu_config = SIMULATION_CONFIGS[trace][rsu_config_name]
    model = VECModel(strategy, rsu_config, DynamicVehicleLoadGenerator(seed=SEED), trace_loader(),
                     load_update_interval=1, seed=SEED)

    grid_qos = defaultdict(list)

    step = 0
    while model.running:
        step += 1
        # if step > 500:
        #     break

        model.step()
        vehicles = model.schedule.get_agents_by_type(VehicleAgent)

        for vehicle in vehicles:
            grid_x = round(vehicle.pos[0])
            grid_y = round(vehicle.pos[1])
            grid_pos = (grid_x, grid_y)
            qos = compute_vehicle_qos(vehicle)
            grid_qos_list = grid_qos[grid_pos]
            grid_qos_list.append((step, qos))

    qos_mean_grid = np.full((200, 200), np.nan)
    qos_min_grid = np.full((200, 200), np.nan)

    for (x, y), qos_list in grid_qos.items():
        qos_mean_grid[y][x] = sum([q[1] for q in qos_list]) / len(qos_list)
        qos_min_grid[y][x] = min([q[1] for q in qos_list])

    filename = f"results_{trace}_{rsu_config_name}_heatmap_qos"
    np.save(filename + "_mean.npy", qos_mean_grid)
    np.save(filename + "_min.npy", qos_min_grid)
    model.datacollector.get_model_vars_dataframe().to_csv("model_vars.csv", index=False)

    plot_qos_grid(trace, rsu_config_name, filename + "_min.npy", min=True)


def plot_qos_grid(trace, rsu_config_name, filename='qos_grid.npy', min=True):
    qos_grid = np.load(filename)
    rsu_config = SIMULATION_CONFIGS[trace][rsu_config_name]

    reduction_factor = 4
    reduced_grid = np.zeros((qos_grid.shape[0] // reduction_factor, qos_grid.shape[1] // reduction_factor))
    for i in range(0, qos_grid.shape[0], reduction_factor):
        for j in range(0, qos_grid.shape[1], reduction_factor):
            slice_ = qos_grid[i:i + reduction_factor, j:j + reduction_factor]
            if np.isnan(slice_).all():
                reduced_grid[i // reduction_factor, j // reduction_factor] = np.nan
            else:
                reduced_grid[i // reduction_factor, j // reduction_factor] = np.nanmin(slice_)

    min_qos_min_value = 0.6 if np.nanmin(reduced_grid) >= 0.6 else 0.15
    assert np.nanmin(reduced_grid) >= min_qos_min_value, \
        f"Error: reduced_grid contains values below the threshold of {min_qos_min_value}."

    print(trace, rsu_config_name, "min MinQoS", np.nanmin(reduced_grid))

    # colors = [(1, 0, 0), (0.9, 0.9, 0.9)]  # Light gray to red
    # cmap = LinearSegmentedColormap.from_list('custom_gray_red', colors, N=256)

    cmap = "plasma"
    label = "Minimum QoS" if min else "Mean QoS"

    # Visualize the numpy array as a heatmap
    fig, ax = plt.subplots()
    heatmap = ax.imshow(reduced_grid, cmap=cmap, interpolation='nearest', vmin=min_qos_min_value, vmax=1)
    plt.colorbar(heatmap, label=label)
    # plt.title(label + " Heatmap")
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.invert_yaxis()

    for i, rsu in enumerate(rsu_config):
        rsu_id = 10001 + i
        pos = (rsu.pos[0] / reduction_factor, rsu.pos[1] / reduction_factor)
        color = VEC_STATION_COLORS[rsu_id]
        ax.add_patch(Rectangle((pos[0] - 0.5, pos[1] - 0.5), 1, 1, facecolor=color))
        ax.add_patch(Circle(pos, rsu.range / reduction_factor, color=color, fill=False, linestyle='--', alpha=1))

    ax.set_aspect('equal')
    plt.tight_layout()
    filename = f"results_{trace}_{rsu_config_name}_{'min' if min else 'avg'}qos_heatmap.png"
    plt.savefig(filename, format="png", dpi=200)
    plt.show()


if __name__ == "__main__":
    logging.basicConfig(level=logging.ERROR)

    # eval_strategy_params()
    # run_all_benchmarks()
    # run_benchmarks("creteil-morning", "4-full")
    investigate_min_qos("creteil-morning", "3-fail-half", DefaultOffloadingStrategy(**BEST_DEFAULT_CONFIG))
    investigate_min_qos("creteil-morning", "3-fail-full", DefaultOffloadingStrategy(**BEST_DEFAULT_CONFIG))
    # plot_qos_grid("creteil-morning", "4-half", "results_creteil-morning_4-half_heatmap_qos_min.npy", min=True)
    # plot_qos_grid("creteil-morning", "9-quarter", "results_creteil-morning_9-quarter_heatmap_qos_min.npy", min=True)
    # plot_qos_grid("qos_grid_min.npy", "Minimum QoS", min=True)
    # plot_qos_versus_vehicle_count()
