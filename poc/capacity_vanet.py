import logging
import math
import time
from abc import ABC, abstractmethod
from functools import lru_cache
from typing import Optional, List

import mesa
import numpy as np
from matplotlib import pyplot as plt, patches
from mesa import Agent, Model
from mesa.space import ContinuousSpace

import VanetTraceLoader as vanetLoader
import simple as simple
from scheduler import RandomActivationBySortedType

SEED = 42

# Constants for the simulation, adjusted for demonstration
TIME_STEP_MS = 500  # Time step in milliseconds
TIME_STEP_S = TIME_STEP_MS / 1000.0  # Time step in seconds
STEPS_PER_SECOND = int(1 // TIME_STEP_S)  # Number of steps per second
assert np.isclose(STEPS_PER_SECOND * TIME_STEP_S, 1.0, rtol=1e-09, atol=1e-09), "Time step conversion error"
VEHICLE_SPEED_FAST_MS = 60 * (1000 / 3600)  # 60 km/h in m/s

VEC_STATION_COLORS = {
    10001: "red",
    10002: "blue",
    10003: "orange",
    10004: "green",
}


class RSAgentStrategy(ABC):
    @abstractmethod
    def handle_offloading(self, station: "VECStationAgent"):
        pass

    def after_step(self, model: "VECModel"):
        # Default implementation does nothing.
        # Subclasses can override this method if needed.
        pass


@lru_cache(maxsize=1)
def get_grid():
    return vanetLoader.get_grid()


@lru_cache(maxsize=1)
def get_traces():
    return vanetLoader.get_traces()


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
    def __init__(self, unique_id, model: "VECModel", trace: Optional[vanetLoader.VehicleTrace]):
        super().__init__(unique_id, model, 0, [])
        self.offloaded_load = 1
        self.invocation = 0
        self.trace_i = 0
        self.trace = trace
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

    @property
    def rsu_distance(self):
        return distance(self.pos, self.station.pos)


def calculate_trajectory_suitability(station: "VECStationAgent", vehicle: VehicleAgent):
    bearing_rad = station.calculate_vehicle_station_bearing(vehicle)
    vehicle_angle_rad = math.radians(vehicle.angle)

    angle_at_vehicle = bearing_rad - vehicle_angle_rad

    ho_metric = (0.5 * math.cos(angle_at_vehicle) + 0.75) / 1.25 * distance(station.pos, vehicle.pos) / station.range

    return ho_metric


def calculate_station_suitability_with_vehicle(station: "VECStationAgent", station_load: float, vehicle: VehicleAgent,
                                               current_station: "VECStationAgent"):
    if station_load + vehicle.offloaded_load > station.capacity:
        return 0

    capacity_suitability = (station.capacity - station_load - vehicle.offloaded_load) / station.capacity
    relative_capacity_suitability = max(10, (current_station.load / current_station.capacity) / (
            station_load / station.capacity)) if station_load > 0 else 10
    trajectory_suitability = calculate_trajectory_suitability(station, vehicle)

    return 0.5 * capacity_suitability + 0.5 * trajectory_suitability + 0.3 * relative_capacity_suitability


class VECStationAgent(simple.VECStationAgent):
    """A VEC station agent with a communication range."""

    # TODO fix inheritance stuff
    def __init__(self, unique_id, model: "VECModel", strategy: RSAgentStrategy, position, range_m, capacity,
                 neighbors=None):
        super().__init__(unique_id, model, 0, 0, 0)
        self.strategy = strategy
        self.pos = position
        self.range = range_m
        self.capacity = capacity
        self.neighbors: List[VECStationAgent] = neighbors if neighbors else []
        self.vehicles: List[VehicleAgent] = []
        self.distance_threshold = 0.7
        self.load_threshold = 0.7
        self.vehicle_distance = None

    @property
    def load(self):
        return sum([vehicle.offloaded_load for vehicle in self.vehicles])

    def get_neighbor_load(self, neighbor_id):
        # noinspection PyTypeChecker
        model: VECModel = self.model
        return model.shared_load_info[neighbor_id]

    def step(self):

        self.strategy.handle_offloading(self)

    def calculate_vehicle_handover_score(self, vehicle: VehicleAgent):
        bearing_rad = self.calculate_vehicle_station_bearing(vehicle)
        vehicle_angle_rad = math.radians(vehicle.angle)

        angle_at_vehicle = bearing_rad - vehicle_angle_rad

        ho_metric = (0.5 * math.cos(angle_at_vehicle) + 0.75) / 1.25 * distance(self.pos, vehicle.pos) / self.range

        return ho_metric

    def calculate_vehicle_station_bearing(self, vehicle: VehicleAgent):
        # Calculate the difference in x and y coordinates
        dx = vehicle.pos[0] - self.pos[0]
        dy = vehicle.pos[1] - self.pos[1]

        # Calculate the angle in radians
        return math.atan2(dy, dx)

    def attempt_handover(self, vehicle: VehicleAgent, force=False) -> bool:

        # current_score = calculate_station_suitability_with_vehicle(self, self.load, vehicle, self)
        neighbors_with_score = [
            (x, calculate_station_suitability_with_vehicle(x, self.get_neighbor_load(x.unique_id), vehicle, self))
            for x in self.neighbors if
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

        # if neighbors_with_score[0][1] < current_score and not force:
        #     logging.info(
        #         f"Vehicle {vehicle.unique_id} is not handed over to any neighbor due to no better suitability (no force)")
        #     return False

        # Loop through sorted neighbors and handover to the first one that accepts
        # TODO this probably doesnt work anymore once we introduce latency
        for neighbor, score in neighbors_with_score:
            success = neighbor.request_handover(vehicle, force)
            if success:
                self.perform_handover(neighbor, vehicle)
                logging.info(f"Vehicle {vehicle.unique_id} handed over to VEC station {neighbor.unique_id}")

                return True

            else:
                # Stats
                # noinspection PyTypeChecker
                model: VECModel = self.model
                model.report_failed_handovers += 1
                model.report_total_failed_handovers += 1

        return False

        # TODO dont do it so dumb
        # For now, take the best one without checking a response
        # self.

        # # Try to find a neighbor station to hand over the vehicle
        # # Sort neighbors by distance to vehicle divided by range
        # # This will prioritize neighbors that are closer to the vehicle and have a larger range
        # neighbors = [(distance(x.pos, vehicle.pos) / x.range, x) for x in self.neighbors if x != self]
        # sorted_neighbors = sorted(neighbors, key=lambda x: x[0])
        #
        # for ratio, neighbor in sorted_neighbors:
        #     # If the ratio is greater than 1, the neighbor is too far away (and so are the rest)
        #     if ratio > 1:
        #         break
        #
        #     if neighbor.load < neighbor.capacity and is_moving_towards(vehicle.pos, vehicle.angle, neighbor.pos):
        #         # Hand over the vehicle to the best neighbor
        #         self.vehicles.remove(vehicle)
        #         neighbor.vehicles.append(vehicle)
        #         vehicle.station = neighbor
        #         print(f"Vehicle {vehicle.unique_id} handed over to VEC station {neighbor.unique_id}")
        #         return
        #
        # # TODO What to do if handover is unavoidable (e.g. capacity is full)?

    def perform_handover(self, to: "VECStationAgent", vehicle: VehicleAgent):
        self.vehicles.remove(vehicle)
        to.vehicles.append(vehicle)
        vehicle.station = to

        # Stats
        # noinspection PyTypeChecker
        model: VECModel = self.model
        model.report_successful_handovers += 1
        model.report_total_successful_handovers += 1

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

    def is_vehicle_exiting(self, vehicle: VehicleAgent):
        # Predict if the vehicle will exit the station's range soon
        # TODO create more precise logic, e.g. based on vehicle speed, distance and range
        # return (distance(self.pos, vehicle.pos) > self.range * self.distance_threshold
        #         and not is_moving_towards(vehicle.pos, vehicle.angle, self.pos))

        return calculate_trajectory_suitability(self, vehicle) > 0.9

    def __repr__(self):
        return f"VECStation{self.unique_id}"


class VECModel(Model):
    """A model with a single vehicle following waypoints on a rectangular road layout."""

    def __init__(self, rs_strategy: RSAgentStrategy, max_capacity=30, load_update_interval=1, start_at=0, **kwargs):
        # Seed is set via super().new()
        super().__init__()
        self.running = True
        self.rs_strategy = rs_strategy
        self.max_capacity = max_capacity
        self.load_update_interval = load_update_interval
        self.traces = get_traces()
        self.width, self.height = vanetLoader.get_size()
        assert self.width == 200
        assert self.height == 200

        self.space = ContinuousSpace(self.width, self.height, False)  # Non-toroidal space
        self.schedule = RandomActivationBySortedType(self, [VehicleAgent, VECStationAgent])

        self.report_successful_handovers = 0
        self.report_failed_handovers = 0
        self.report_total_successful_handovers = 0
        self.report_total_failed_handovers = 0

        self.shared_load_info = {}

        def vehicle_count_collector(a: Agent):
            if isinstance(a, VECStationAgent):
                return len(a.vehicles)
            return None

        self.datacollector = mesa.DataCollector(
            model_reporters={
                "SuccessfulHandoverCount": "report_successful_handovers",
                "TotalSuccessfulHandoverCount": "report_total_successful_handovers",
                "FailedHandoverCount": "report_failed_handovers",
                "TotalFailedHandoverCount": "report_total_failed_handovers",
                "AvgQoS": VECModel.report_avg_qos,
                "MinQoS": VECModel.report_min_qos,
                "GiniLoad": VECModel.report_gini_load,
            },
            agent_reporters={"Distances": "vehicle_distance", "StationVehicleCount": vehicle_count_collector}
        )

        # station_positions = [
        #     (75, 50),
        #     (50, 115),
        #     (110, 140),
        #     (140, 60)
        # ]
        # TODO check station positions
        station_positions = [
            (75, 45),  # red
            (40, 120),  # blue
            (120, 155),  # yellow
            (160, 50)  # green
        ]
        self.vec_stations = []
        for i, pos in enumerate(station_positions, start=1):
            station = VECStationAgent(10000 + i, self, self.rs_strategy, pos, 65, max_capacity)
            self.vec_stations.append(station)
            self.schedule.add(station)

        for i in range(4):
            self.vec_stations[i].neighbors = [station for station in self.vec_stations
                                              if distance(station.pos, self.vec_stations[i].pos) <= station.range +
                                              self.vec_stations[i].range and station != self.vec_stations[i]]

        self.vehicle_id = 1

        self.unplaced_vehicles: List[vanetLoader.VehicleTrace] = [v for k, v in self.traces.items()]
        self.unplaced_vehicles.sort(key=lambda x: x.first_ts, reverse=True)

        # ONLY DEBUG
        # self.unplaced_vehicles = self.unplaced_vehicles[:1]
        # self.unplaced_vehicles = list(filter(lambda x: x.id == 'VehicleFlowEastToNorth.0', self.unplaced_vehicles))

        self.to_remove: List[VehicleAgent] = []

        self.vehicle = None

        self.step_second = 0

        self.datacollector.collect(self)

        for _ in range(start_at):
            self.step()

    def spawn_vehicle(self, trace_id, step):
        vehicle = VehicleAgent(self.vehicle_id, self, self.traces[trace_id])
        vehicle.ts = step // TIME_STEP_S

        self.schedule.add(vehicle)

        self.vehicle_id += 1

        station = min(self.vec_stations, key=lambda x: distance(x.pos, vehicle.pos))
        station.vehicles.append(vehicle)
        vehicle.station = station

        if vehicle.unique_id == 265:
            self.vehicle = vehicle

        return vehicle

    def step(self):

        while self.to_remove and self.to_remove[-1].trace.last_ts == self.step_second:
            v = self.to_remove.pop()
            v.station.vehicles.remove(v)
            self.schedule.remove(v)
            v.remove()

        assert len(self.to_remove) == len(self.agents) - 4, "Agent count mismatch"  # 4 is number of stations
        self.step_second += 1

        while self.unplaced_vehicles and self.unplaced_vehicles[-1].first_ts == self.step_second:
            v_trace = self.unplaced_vehicles.pop()
            v = self.spawn_vehicle(v_trace.id, self.step_second)

            # Insert while sorting on last_ts
            self.to_remove.append(v)
            self.to_remove.sort(key=lambda x: x.trace.last_ts, reverse=True)

        # TODO simplify??
        for _ in range(STEPS_PER_SECOND):
            self.schedule.step()
            self.rs_strategy.after_step(self)

        self.datacollector.collect(self)

        if self.step_second % self.load_update_interval == 0:
            self.update_shared_load_info()

        # Reset per-step statistics
        self.report_successful_handovers = 0
        self.report_failed_handovers = 0

    def update_shared_load_info(self):
        self.shared_load_info = {station.unique_id: station.load for station in self.vec_stations}

    def report_avg_qos(self):
        qos = compute_qos(self)
        if len(qos) == 0:
            return 1
        return sum(qos) / len(qos)

    def report_min_qos(self):
        qos = compute_qos(self)
        return min(qos, default=1)

    def report_gini_load(self):
        loads = [station.load for station in self.vec_stations]
        return compute_gini(loads)


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


def compute_qos(model: VECModel) -> List[float]:
    """
    Compute QoS as the load of the station divided by its capacity if the vehicle is within range, otherwise 0.
    """

    def qos_internal(agent: VehicleAgent):
        distance_factor = 1
        dist = distance(agent.pos, agent.station.pos)
        # TODO check (Simplification) Distance related QoS decreases linearly
        if agent.station.range < dist <= 2 * agent.station.range:
            distance_factor = 1 - (dist - agent.station.range) / agent.station.range
        elif dist > 2 * agent.station.range:
            distance_factor = 0

        load_factor = 1
        if agent.station.load > agent.station.capacity:
            load_factor = agent.station.capacity / agent.station.load

        return load_factor * distance_factor

    qos_list = []
    for agent in model.schedule.get_agents_by_type(VehicleAgent):
        qos = qos_internal(agent)
        qos_list.append(qos)

    return qos_list


class DefaultOffloadingStrategy(RSAgentStrategy):
    def handle_offloading(self, station: VECStationAgent):
        # Only temporary for demonstration
        # self.vehicle_distance = distance(self.pos, self.model.vehicle.pos)

        # Hand-over vehicles that are leaving anyways (todo remove in later iteration??)
        for vehicle in list(station.vehicles):
            # Check if vehicle is exiting the station's range soon
            if calculate_trajectory_suitability(station, vehicle) < 0.9:
                continue

            logging.debug(f"Vehicle {vehicle.unique_id} is leaving the station {station.unique_id} range")
            success = station.attempt_handover(vehicle)

            if not success and calculate_trajectory_suitability(station, vehicle) > 0.95:
                # Force handover
                logging.info(f"Vehicle {vehicle.unique_id} is being forced to leave the station {station.unique_id}")
                station.attempt_handover(vehicle, force=True)

        # TODO move to global
        # TODO should also consider other stations
        if station.load < station.load_threshold * station.capacity:
            return

        # Iterate through vehicles sorted by
        for vehicle in sorted(station.vehicles, key=lambda x: station.calculate_vehicle_handover_score(x),
                              reverse=True):
            # TODO move to global
            # TODO should also consider other stations
            if station.load < station.load_threshold * station.capacity:
                return

            logging.info(f"Vehicle {vehicle.unique_id} is being considered for handover due to overload")

            station.attempt_handover(vehicle, force=station.load > 0.95 * station.capacity)


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
                nearest_station = min(station.neighbors, key=lambda x: distance(x.pos, vehicle.pos))
                if nearest_station == station:
                    logging.warning(f"Vehicle {vehicle.unique_id} is out of range of all RSUs")
                    continue
                logging.info(
                    f"Vehicle {vehicle.unique_id} is being handed over to the nearest station {nearest_station.unique_id}")
                station.perform_handover(nearest_station, vehicle)


def main():
    # Configuration for demonstration
    road_width = 200  # meters
    road_height = 200  # meters
    vehicle_speed = VEHICLE_SPEED_FAST_MS

    # Initialize and run the model
    model = VECModel(road_width, road_height, vehicle_speed)
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


def print_model_metrics(model, model_name):
    """
    Prints the evaluation metrics for a given model.

    Parameters:
    - model: The model object to extract metrics from.
    - model_name: A string representing the name or identifier of the model.
    """
    start_index = 100
    df = model.datacollector.get_model_vars_dataframe()
    print(f"{model_name} Success: {df['TotalSuccessfulHandoverCount'].iloc[-1]}")
    print(f"{model_name} Failed: {df['TotalFailedHandoverCount'].iloc[-1]}")
    print(f"{model_name} QoS: {df['AvgQoS'][start_index:].mean()}")
    print(f"{model_name} QoSMin: {df['MinQoS'][start_index:].mean()}")
    print(f"{model_name} Gini: {df['GiniLoad'][start_index:].mean()}")


def compare_load_sharing():
    start = time.time()

    rs_strategy = DefaultOffloadingStrategy()

    model1 = VECModel(rs_strategy, 25, 1, seed=SEED)
    model5 = VECModel(rs_strategy, 25, 5, seed=SEED)
    model10 = VECModel(rs_strategy, 25, 10, seed=SEED)

    for i in range(1000):
        if (i + 1) % 100 == 0:
            print(i + 1)
        model1.step()
        model5.step()
        model10.step()

    # params = {
    #     "load_update_interval": [1, 5, 10],
    #     "max_capacity": 25,
    #     "seed": SEED
    # }
    #
    # results = mesa.batch_run(VECModel, params, max_steps=1000, number_processes=None)

    print("Time elapsed:", int(time.time() - start), "s")

    # "Regression test"
    print_model_metrics(model1, "ShareLoadFreq1")
    print_model_metrics(model5, "ShareLoadFreq5")
    print_model_metrics(model10, "ShareLoadFreq10")

    assert model1.report_total_successful_handovers == 2959
    assert model5.report_total_successful_handovers == 2905
    assert model10.report_total_successful_handovers == 2751


if __name__ == "__main__":
    logging.basicConfig(level=logging.ERROR)

    # main()
    compare_load_sharing()
