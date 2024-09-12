import math
from abc import ABC, abstractmethod
from typing import Optional, List, Dict

import mesa
import numpy as np
from mesa import Agent, Model
from mesa.space import ContinuousSpace

from poc import VanetTraceLoader as vanetLoader
from poc.VanetTraceLoader import VehicleTrace
from poc.base import RsuConfig, distance
from poc.scheduler import RandomActivationBySortedType


class RSAgentStrategy(ABC):
    """
    A strategy for handling offloading in a VEC station.
    """

    @abstractmethod
    def handle_offloading(self, station: "VECStationAgent"):
        pass

    def after_step(self, model: "VECModel"):
        # Default implementation does nothing.
        # Subclasses can override this method if needed.
        pass


class VehicleLoadGenerator(ABC):
    """
    A generator for vehicle loads.
    """

    @abstractmethod
    def compute_offloaded_load(self, vehicle: "VehicleAgent"):
        """
        Compute the offloaded load of a vehicle.
        """
        pass


class VehicleAgent(Agent):
    """A vehicle agent that follows a list of waypoints and calculates its angle."""

    def __init__(self, unique_id, model: "VECModel", trace: Optional[vanetLoader.VehicleTrace],
                 load_gen: VehicleLoadGenerator, steps_per_second):
        super().__init__(unique_id, model)
        self.offloaded_load = 0
        self.invocation = 0
        self.trace_i = 0
        self.trace = trace
        self.load_gen = load_gen
        self.steps_per_second = steps_per_second
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
        """
        Perform a single step of the vehicle agent.
        """

        _, state = next(self.trace_iterator)
        assert state['timestep_time'] == self.trace_i + self.trace.first_ts, "Time step mismatch"
        self.trace_i += 1

        self.pos = (state['vehicle_x'], state['vehicle_y'])
        self.angle = state['vehicle_angle']
        self.model.space.move_agent(self, self.pos)

        self.offloaded_load = self.load_gen.compute_offloaded_load(self)

    def step(self):
        """
        Perform a single step of the vehicle agent.
        """

        if self.invocation % self.steps_per_second == 0:
            self.do_step()

        self.invocation += 1

    def count_nearby_vehicles(self) -> int:
        """
        Count the number of nearby vehicles.
        """

        count = 0
        nearby_dist = 5
        for agent in self.model.schedule.get_agents_by_type(VehicleAgent):
            if agent != self and distance(agent.pos, self.pos) <= nearby_dist:
                count += 1

        return count

    @property
    def rsu_distance(self):
        """
        Calculate the distance to the station.
        """

        return distance(self.pos, self.station.pos)

    def __repr__(self):
        return f"VehicleAgent{self.unique_id}"


class VECStationAgent(Agent):
    """A VEC station agent with a communication range."""

    def __init__(self, unique_id, model: "VECModel", strategy: RSAgentStrategy, position, range_m, capacity,
                 neighbors=None, can_read_neighbor_load=False):
        super().__init__(unique_id, model)
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
        """
        Calculate the load of the station.
        """

        return sum([vehicle.offloaded_load for vehicle in self.vehicles])

    @property
    def utilization(self):
        """
        Calculate the utilization of the station.
        """

        return self.load / self.capacity

    @property
    def connected_vehicles(self):
        """
        Calculate the number of connected vehicles.
        """

        return len(self.vehicles)

    def get_neighbor_load(self, neighbor_id):
        """
        Get the load of a neighbor station.
        """

        if self.can_read_neighbor_load:
            return [x.load for x in self.neighbors if x.unique_id == neighbor_id][0]
        return self.neighbor_load[neighbor_id]

    def increment_neighbor_load(self, neighbor_id, load):
        """
        Increment the load of a neighbor station.
        """

        self.neighbor_load[neighbor_id] += load
        # Might be less than 0 because of delayed load sharing
        if self.neighbor_load[neighbor_id] < 0:
            self.neighbor_load[neighbor_id] = 0

    def step(self):
        """
        Perform a single step of the VEC station agent.
        """

        self.strategy.handle_offloading(self)

    def calculate_vehicle_station_bearing(self, vehicle: VehicleAgent):
        """
        Calculate the angle between the vehicle and the station.
        """

        dx = vehicle.pos[0] - self.pos[0]
        dy = vehicle.pos[1] - self.pos[1]

        return math.atan2(dy, dx)

    def perform_handover(self, to: "VECStationAgent", vehicle: VehicleAgent, cause="range"):
        """
        Perform a handover of a vehicle to another station.
        """

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
        """
        Report a failed handover.
        """

        # noinspection PyTypeChecker
        model: VECModel = self.model
        model.report_failed_handovers += 1
        model.report_total_failed_handovers += 1

    def request_handover(self, vehicle: VehicleAgent, force=False) -> bool:
        """
        A station requests that a vehicle be handed over to a neighbor station.
        The target station can accept or reject the request based on its own criteria.
        """

        if not force:
            # Check if the vehicle is 1. in range and 2. the RSU has enough capacity
            if distance(self.pos, vehicle.pos) > self.range:
                return False
            if self.load + vehicle.offloaded_load > self.capacity:
                return False

        return True

    def __repr__(self):
        return f"VECStation{self.unique_id}"

    def is_vehicle_in_range(self, vehicle: VehicleAgent):
        """
        Check if a vehicle is within the station's range.
        """

        return distance(self.pos, vehicle.pos) <= self.range


class VECModel(Model):
    """A model with a single vehicle following waypoints on a rectangular road layout."""

    def __init__(self, rs_strategy: RSAgentStrategy, rsu_configs: List[RsuConfig],
                 vehicle_load_gen: VehicleLoadGenerator, traces: Dict[str, VehicleTrace], steps_per_second,
                 load_update_interval=1, start_at=0, **kwargs):
        # Seed is set via super().new()
        super().__init__()
        self.running = True
        self.rs_strategy = rs_strategy
        self.load_update_interval = load_update_interval
        self.traces = traces
        self.steps_per_second = steps_per_second
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
        """
        Spawn a vehicle at the given step.
        """

        self.vehicle_id += 1
        vehicle = VehicleAgent(self.vehicle_id, self, self.traces[trace_id], self.vehicle_load_gen,
                               self.steps_per_second)
        vehicle.ts = step // (1 / self.steps_per_second)

        self.schedule.add(vehicle)

        station = min(self.vec_stations, key=lambda x: distance(x.pos, vehicle.pos))
        station.vehicles.append(vehicle)
        vehicle.station = station

        return vehicle

    def step(self):
        """
        Perform a single step of the model.
        """

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

        for _ in range(self.steps_per_second):
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
        """
        Update the shared load information between stations.
        """

        for station in self.vec_stations:
            for neighbor in self.vec_stations:
                if neighbor in station.neighbor_load or neighbor in station.neighbors:
                    station.neighbor_load[neighbor.unique_id] = neighbor.load

    def report_avg_qos(self):
        """
        Report the average QoS.
        """

        qos = compute_qos(self)
        if len(qos) == 0:
            return 1
        result = sum(qos) / len(qos)
        other_qos = [x * y for x, y in zip(compute_load_based_qos(self), compute_range_based_qos(self))]
        other_result = sum(other_qos) / len(other_qos)
        assert abs(result - other_result) < 1e-9, f"QoS mismatch: {result} vs {other_result}"
        return sum(qos) / len(qos)

    def report_min_qos(self):
        """
        Report the minimum QoS.
        """

        qos = compute_qos(self)
        return min(qos, default=1)

    def report_avg_qos_load_based(self):
        """
        Report the average load-based QoS.
        """

        qos = compute_load_based_qos(self)
        if len(qos) == 0:
            return 1
        return sum(qos) / len(qos)

    def report_min_qos_load_based(self):
        """
        Report the minimum load-based QoS.
        """

        qos = compute_load_based_qos(self)
        return min(qos, default=1)

    def report_avg_qos_range_based(self):
        """
        Report the average range-based QoS.
        """

        qos = compute_range_based_qos(self)
        if len(qos) == 0:
            return 1
        return sum(qos) / len(qos)

    def report_min_qos_range_based(self):
        """
        Report the minimum range-based QoS.
        """

        qos = compute_range_based_qos(self)
        return min(qos, default=1)

    def report_gini_load(self):
        """
        Report the Gini coefficient of the station loads.
        """

        loads = [station.load for station in self.vec_stations]
        return compute_gini(loads)

    def report_vehicle_count(self):
        """
        Report the number of vehicles.
        """

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
