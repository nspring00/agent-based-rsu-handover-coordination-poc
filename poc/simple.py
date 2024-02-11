from matplotlib import pyplot as plt, patches
from mesa import Agent, Model
from mesa.time import BaseScheduler
from mesa.space import ContinuousSpace
import numpy as np

# Constants for the simulation, adjusted for demonstration
TIME_STEP_MS = 50  # Time step in milliseconds
TIME_STEP_S = TIME_STEP_MS / 1000.0  # Time step in seconds
VEHICLE_SPEED_FAST_MS = 60 * (1000 / 3600)  # 60 km/h in m/s


class VehicleAgent(Agent):
    """A vehicle agent that follows a list of waypoints and calculates its angle."""

    def __init__(self, unique_id, model, speed, waypoints):
        super().__init__(unique_id, model)
        self.speed = speed
        self.waypoints = waypoints
        self.current_waypoint_index = 0
        self.angle = 0.0  # Angle in degrees

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

    def __init__(self, unique_id, model, position, range_m):
        super().__init__(unique_id, model)
        self.pos = position
        self.range = range_m


class VECModel(Model):
    """A model with a single vehicle following waypoints on a rectangular road layout."""

    def __init__(self, width, height, speed):
        self.width = width
        self.height = height

        self.space = ContinuousSpace(width, height, False)  # Non-toroidal space
        self.schedule = BaseScheduler(self)

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
            self.agents.append(station)

        # Create a vehicle with the defined waypoints
        vehicle = VehicleAgent(1, self, speed, waypoints)
        self.vehicle = vehicle
        self.schedule.add(vehicle)
        self.agents.append(vehicle)

        # Place the vehicle at the first waypoint
        self.space.place_agent(vehicle, waypoints[0])

    def step(self):
        self.schedule.step()


# Configuration for demonstration
road_width = 100  # meters
road_height = 80  # meters
vehicle_speed = VEHICLE_SPEED_FAST_MS

# Initialize and run the model
model = VECModel(road_width, road_height, vehicle_speed)
output = []

vehicle_positions = []  # For recording vehicle positions
vehicle_angles = []  # For recording vehicle angles

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
range_line_style = 'r:'  # Red dotted line for the range circle

for station in model.vec_stations:
    # Draw the station as a square
    lower_left_corner = (station.pos[0] - station_size / 2, station.pos[1] - station_size / 2)
    square = patches.Rectangle(lower_left_corner, station_size, station_size, color='red')
    ax.add_patch(square)

    # Draw the range as a dotted circle
    circle = plt.Circle(station.pos, station.range, color='red', fill=False, linestyle='dotted')
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
    pass
