import math
from typing import Optional

import mesa
import solara
from matplotlib import pyplot as plt, patches
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle, Circle
from mesa import Agent, Model
from mesa.time import BaseScheduler
from mesa.space import ContinuousSpace, FloatCoordinate
import numpy as np

# Constants for the simulation, adjusted for demonstration
TIME_STEP_MS = 50  # Time step in milliseconds
TIME_STEP_S = TIME_STEP_MS / 1000.0  # Time step in seconds
VEHICLE_SPEED_FAST_MS = 60 * (1000 / 3600)  # 60 km/h in m/s

VEC_STATION_COLORS = {
    10001: "red",
    10002: "blue",
    10003: "orange",
    10004: "green",
}


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


class VehicleAgent(Agent):
    """A vehicle agent that follows a list of waypoints and calculates its angle."""

    def __init__(self, unique_id, model, speed, waypoints):
        super().__init__(unique_id, model)
        self.speed = speed
        self.waypoints = waypoints
        self.current_waypoint_index = 0
        self.angle = 0.0  # Angle in degrees
        self.station: Optional["VECStationAgent"] = None

    def move_towards_waypoint(self):
        """Move the agent towards the current waypoint."""
        current_waypoint = self.waypoints[self.current_waypoint_index]
        x, y = self.pos
        waypoint_x, waypoint_y = current_waypoint

        # Calculate direction vector and distance to waypoint
        direction_vector = np.array([waypoint_x - x, waypoint_y - y])
        distance_to_waypoint = np.linalg.norm(direction_vector)
        direction_vector_normalized = direction_vector / distance_to_waypoint if distance_to_waypoint else direction_vector

        # Calculate step size
        step_size = min(self.speed * TIME_STEP_S, distance_to_waypoint)

        # Update position
        new_x, new_y = np.array(self.pos) + direction_vector_normalized * step_size
        self.model.space.move_agent(self, (new_x, new_y))

        # Update angle
        if distance_to_waypoint:
            self.angle = np.degrees(np.arctan2(direction_vector_normalized[1], direction_vector_normalized[0])) % 360

        # Check if waypoint is reached and update waypoint index
        if distance_to_waypoint <= step_size:
            self.current_waypoint_index = (self.current_waypoint_index + 1) % len(self.waypoints)

    def step(self):
        self.move_towards_waypoint()


class VECStationAgent(Agent):
    """A VEC station agent with a communication range."""

    def __init__(self, unique_id, model, position, range_m, neighbors=None):
        super().__init__(unique_id, model)
        self.pos = position
        self.range = range_m
        self.neighbors = neighbors if neighbors else []
        self.vehicles = []
        self.threshold = 0.7
        self.vehicle_distance = None

    def step(self):

        demo_vehicle = self.model.vehicle
        self.vehicle_distance = distance(self.pos, demo_vehicle.pos)

        for vehicle in self.vehicles:
            # Check if vehicle is moving towards the station
            if is_moving_towards(vehicle.pos, vehicle.angle, self.pos):
                continue
            # Check if vehicle is within the handover threshold
            dist = distance(self.pos, vehicle.pos)
            if dist > self.threshold * self.range:
                # Try to find a neighbor station to hand over the vehicle
                # Sort neighbors by distance to vehicle divided by range
                # This will prioritize neighbors that are closer to the vehicle and have a larger range
                neighbors = [(distance(x.pos, vehicle.pos) / x.range, x) for x in self.neighbors if x != self]
                sorted_neighbors = sorted(neighbors, key=lambda x: x[0])
                for ratio, neighbor in sorted_neighbors:
                    # If the ratio is greater than 1, the neighbor is too far away
                    if ratio > 1:
                        break
                    # Check if the vehicle is moving towards the neighbor
                    if not is_moving_towards(vehicle.pos, vehicle.angle, neighbor.pos):
                        continue

                    # Hand over the vehicle to the best neighbor
                    self.vehicles.remove(vehicle)
                    neighbor.vehicles.append(vehicle)
                    vehicle.station = neighbor
                    print(f"Vehicle {vehicle.unique_id} handed over to VEC station {neighbor.unique_id}")

    def __repr__(self):
        return f"VECStation{self.unique_id}"


class VECModel(Model):
    """A model with a single vehicle following waypoints on a rectangular road layout."""

    def __init__(self, width, height, speed):
        self.width = width
        self.height = height

        self.space = ContinuousSpace(width, height, False)  # Non-toroidal space
        self.schedule = BaseScheduler(self)

        def vehicle_count_collector(a: Agent):
            if isinstance(a, VECStationAgent):
                return len(a.vehicles)
            return None

        self.datacollector = mesa.DataCollector(
            agent_reporters={"Distances": "vehicle_distance", "StationVehicleCount": vehicle_count_collector}
        )

        self.running = True

        self.agents = []

        # Define waypoints at the corners of the rectangular road layout
        waypoints_pos_offset = 5
        waypoints = [(waypoints_pos_offset, waypoints_pos_offset), (width - waypoints_pos_offset, waypoints_pos_offset),
                     (width - waypoints_pos_offset, height - waypoints_pos_offset),
                     (waypoints_pos_offset, height - waypoints_pos_offset)]
        self.waypoints = waypoints

        station_pos_offset = 5
        station_positions = [
            (waypoints[0][0] + station_pos_offset, waypoints[0][1] + station_pos_offset),
            (waypoints[1][0] - station_pos_offset, waypoints[1][1] + station_pos_offset),
            (waypoints[2][0] - station_pos_offset, waypoints[2][1] - station_pos_offset),
            (waypoints[3][0] + station_pos_offset, waypoints[3][1] - station_pos_offset)
        ]
        self.vec_stations = []
        for i, pos in enumerate(station_positions, start=1):
            station = VECStationAgent(10000 + i, self, pos, 45)
            self.vec_stations.append(station)
            self.space.place_agent(station, pos)
            self.schedule.add(station)
            self.agents.append(station)

        for i in range(4):
            self.vec_stations[i].neighbors = [self.vec_stations[(i + 1) % 4], self.vec_stations[(i + 3) % 4]]

        vehicle_id = 1

        def spawn_vehicle(pos: FloatCoordinate, angle, wp_offset, reverse=False):
            nonlocal vehicle_id
            wps = self.waypoints[::-1] if reverse else self.waypoints
            vehicle = VehicleAgent(vehicle_id, self, speed, wps)
            vehicle.angle = angle
            vehicle.current_waypoint_index = wp_offset
            self.schedule.add(vehicle)
            self.agents.append(vehicle)

            self.space.place_agent(vehicle, pos)
            vehicle_id += 1

            station = min(self.vec_stations, key=lambda x: distance(x.pos, vehicle.pos))
            station.vehicles.append(vehicle)
            vehicle.station = station

            if vehicle_id == 2:
                self.vehicle = vehicle
                for station in self.vec_stations:
                    station.vehicle_distance = distance(station.pos, vehicle.pos)

        spawn_vehicle(waypoints[0], 0, 0)
        spawn_vehicle((waypoints[1][0], waypoints[1][1] + 50), 90, 2)
        spawn_vehicle((waypoints[3][0] + 20, waypoints[3][1]), 0, 1, True)

        self.datacollector.collect(self)

    def step(self):
        self.schedule.step()
        self.datacollector.collect(self)


def agent_portrayal(agent):
    # Render vehicle agent
    if isinstance(agent, VehicleAgent):
        return {
            "color": VEC_STATION_COLORS[agent.station.unique_id],
            "s": 50,
        }

    # Render VEC station agent
    if isinstance(agent, VECStationAgent):
        portrayal = {
            "shape": "rect",
            "color": VEC_STATION_COLORS[agent.unique_id],
            "filled": "true",
            "layer": 0,
            "w": 1,
            "h": 1,
        }
        return portrayal

    assert False


def render_model(model):
    fig = Figure()
    ax = fig.subplots()

    min_x = min(waypoint[0] for waypoint in model.waypoints)
    max_x = max(waypoint[0] for waypoint in model.waypoints)
    min_y = min(waypoint[1] for waypoint in model.waypoints)
    max_y = max(waypoint[1] for waypoint in model.waypoints)
    road_width = max_x - min_x
    road_height = max_y - min_y

    # Draw the road
    road = Rectangle((min_x, min_y), road_width, road_height, linewidth=3, edgecolor='gray', facecolor='none')
    ax.add_patch(road)

    # Draw all agents
    for agent in model.agents:
        if isinstance(agent, VehicleAgent):
            ax.add_patch(Circle(agent.pos, 3, facecolor=VEC_STATION_COLORS[agent.station.unique_id], edgecolor='black'))
            ax.text(agent.pos[0], agent.pos[1], str(agent.unique_id), ha='center', va='center')
        elif isinstance(agent, VECStationAgent):
            color = VEC_STATION_COLORS[agent.unique_id]
            ax.add_patch(Rectangle((agent.pos[0] - 3, agent.pos[1] - 3), 6, 6, facecolor=color))
            range_circle = Circle(agent.pos, agent.range, color=color, fill=False, linestyle='--')
            ax.add_patch(range_circle)
            range_circle = Circle(agent.pos, agent.threshold * agent.range, color=color, fill=False, linestyle='--',
                                  alpha=0.5)
            ax.add_patch(range_circle)

    ax.set_xlim(0, model.width)
    ax.set_ylim(0, model.height)

    solara.FigureMatplotlib(fig)


def render_distance_chart(model: VECModel):
    fig = Figure()
    ax = fig.subplots()

    data = model.datacollector.get_agent_vars_dataframe()['Distances']
    filtered_distances = data.loc[data.index.get_level_values('AgentID') >= 10000]
    df = filtered_distances.unstack(level="AgentID")

    for station_id, color in VEC_STATION_COLORS.items():
        assert station_id in df.columns
        df[station_id].plot(ax=ax, color=color)

    ax.set_title('Distances from VEC stations')
    ax.set_xlabel('Step')
    ax.set_ylabel('Distance')
    solara.FigureMatplotlib(fig)


def render_station_vehicle_count_chart(model: VECModel):
    fig = Figure()
    ax = fig.subplots()

    data = model.datacollector.get_agent_vars_dataframe()['StationVehicleCount']
    filtered_counts = data.loc[data.index.get_level_values('AgentID') >= 10000]
    df = filtered_counts.unstack(level="AgentID")

    for station_id, color in VEC_STATION_COLORS.items():
        assert station_id in df.columns
        df[station_id].plot(ax=ax, color=color)

    ax.set_title('Vehicle count at VEC stations')
    ax.set_xlabel('Step')
    ax.set_ylabel('Vehicle count')
    solara.FigureMatplotlib(fig)


def main():
    # Configuration for demonstration
    road_width = 100  # meters
    road_height = 80  # meters
    vehicle_speed = VEHICLE_SPEED_FAST_MS

    # Initialize and run the model
    model = VECModel(road_width, road_height, vehicle_speed)
    output = []

    vehicle_positions = []  # For recording vehicle positions
    vehicle_angles = []  # For recording vehicle angles

    render_distance_chart(model)

    # Run the simulation for 200 steps to observe the vehicle's movement and rotation
    for i in range(400):
        model.step()
        vehicle = model.vehicle
        if i % 20 == 0:  # Collect position and angle for every 20 steps
            output.append(f"Step {i}: Vehicle Position: {vehicle.pos}, Angle: {vehicle.angle:.2f}")
            vehicle_positions.append(vehicle.pos)
            vehicle_angles.append(vehicle.angle)

    print(*output, sep='\n')

    # Visualize the recorded vehicle positions and directions
    fig, ax = plt.subplots()
    road = patches.Polygon(model.waypoints, closed=True, fill=False, edgecolor='black', linewidth=2)
    ax.add_patch(road)

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
    vehicle_length = 2  # Length of the vehicle arrow

    for position, angle in zip(vehicle_positions, vehicle_angles):
        x, y = position
        dx = vehicle_length * np.cos(np.radians(angle))
        dy = vehicle_length * np.sin(np.radians(angle))
        ax.arrow(x, y, dx, dy, head_width=1, head_length=1, fc='blue', ec='blue')

    ax.set_xlim(0, road_width)
    ax.set_ylim(0, road_height)
    # ax.set_aspect('equal')
    plt.xlabel('Distance (meters)')
    plt.ylabel('Distance (meters)')
    plt.title('Recorded Vehicle Movement on a Rectangular Road')
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()


def test_moving_towards():
    assert is_moving_towards((0, 0), 45, (1, 1)), "Vehicle should be moving towards the station"


def test_moving_away():
    assert not is_moving_towards((0, 0), 225, (1, 1)), "Vehicle should be moving away from the station"


def test_moving_orthogonal():
    assert not is_moving_towards((0, 0), 136, (1, 1)), "Vehicle should be moving orthogonal to the station"
    assert not is_moving_towards((0, 0), 314, (1, 1)), "Vehicle should be moving orthogonal to the station"


def test_moving_directly_towards():
    assert is_moving_towards((0, 0), 0, (10, 0)), "Vehicle should be moving directly towards the station on the x-axis"


def test_moving_directly_away():
    assert not is_moving_towards((10, 0), 180,
                                 (20, 0)), "Vehicle should be moving directly away from the station on the x-axis"
