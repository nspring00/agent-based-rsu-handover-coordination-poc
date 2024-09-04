import itertools
import logging
import math
from collections import defaultdict, deque

import numpy as np

from poc import units as units
from poc.base import distance
from poc.model import VehicleLoadGenerator, VehicleAgent, RSAgentStrategy, VECStationAgent, VECModel


class StaticVehicleLoadGenerator(VehicleLoadGenerator):
    def compute_offloaded_load(self, vehicle: "VehicleAgent"):
        return 1


class DefaultOffloadingStrategy(RSAgentStrategy):
    def __init__(self, overload_threshold=0.95, leaving_threshold=0.05, lb_util_hysteresis=0.1,
                 alt_suitability_min=0.2):
        self.overload_threshold = overload_threshold
        self.leaving_threshold = leaving_threshold
        self.lb_util_hysteresis = lb_util_hysteresis
        self.alt_suitability_min = alt_suitability_min

    def handle_offloading(self, station: VECStationAgent):

        # Step 1: Hand-over vehicles that are leaving anyway
        self.handle_vehicle_leaving_range(station)

        # Step 2: Hand-over vehicles to neighboring stations considering load balancing (also handles overload)
        self.handle_load_balancing_with_neighbors(station)

    def handle_vehicle_leaving_range(self, station):
        """
        Handle vehicles that are leaving the station's range.
        Iterate through vehicles and hand over if the trajectory suitability is below a certain threshold.
        If the suitability is below the leaving threshold, the handover is forced.
        """

        for vehicle in list(station.vehicles):
            # Based on trajectory suitability, decide if the vehicle should be handed over
            trajectory_suitability = calculate_trajectory_suitability(station, vehicle)
            if trajectory_suitability <= self.leaving_threshold:
                self.attempt_handover_vehicle(station, vehicle, "range", force=not station.is_vehicle_in_range(vehicle))

    def handle_load_balancing_with_neighbors(self, current: VECStationAgent):
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
            if neighbor_utilization > current.utilization - self.lb_util_hysteresis:
                break

            success = neighbor_station.request_handover(vehicle, force=is_overload)
            if not success:
                current.report_failed_handover()
                continue

            logging.info(
                f"Vehicle {vehicle.unique_id} is being handed over to VEC station {neighbor_station.unique_id} to balance load")
            current.perform_handover(neighbor_station, vehicle, "overload" if is_overload else "load_balancing")

            # Recursive call to perform potentially multiple load-balancing related handovers
            self.handle_load_balancing_with_neighbors(current)
            break

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
            in_range_stations = [x for x in station.neighbors
                                 if x.unique_id not in self.previously_connected[vehicle.unique_id]
                                 and distance(x.pos, vehicle.pos) <= x.range
                                 and is_moving_towards(vehicle.pos, vehicle.angle, x.pos)]

            if not in_range_stations:
                if station.is_vehicle_in_range(vehicle):
                    continue

                # Special case: All stations in range are in previous connections
                in_range_stations = [x for x in station.neighbors
                                     if distance(x.pos, vehicle.pos) <= x.range]

                if not in_range_stations:
                    logging.warning(f"Vehicle {vehicle.unique_id} is out of range of all RSUs")
                    continue

            # Get the closest station that wasn't previously connected
            nearest_station = min(in_range_stations, key=lambda x: distance(x.pos, vehicle.pos))

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
            in_range_stations = [x for x in station.neighbors
                                 if x.unique_id not in self.previously_connected[vehicle.unique_id]
                                 and distance(x.pos, vehicle.pos) <= x.range]

            if not in_range_stations:
                continue

            # Get the closest station that wasn't previously connected
            nearest_station = min(in_range_stations, key=lambda x: distance(x.pos, vehicle.pos))

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


STRATEGIES_DICT = {
    # "default": DefaultOffloadingStrategy,
    "default": DefaultOffloadingStrategy,
    "nearest": NearestRSUStrategy,
    "earliest": EarliestPossibleHandoverStrategy,
    "earliest2": EarliestPossibleHandoverNoBackStrategy,
    "latest": LatestPossibleHandoverStrategy,
}
