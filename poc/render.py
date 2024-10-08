import math

import solara
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle, Circle, FancyArrow

from poc.model import VehicleAgent, VECStationAgent, VECModel


def render_model_with_bg(background):
    """
    Create a render function that renders the model with a background image.
    """

    return lambda model: render_model(model, background)


def render_model(model: VECModel, background=None):
    """
    Render the model with the given background image.
    """

    fig = Figure()
    ax = fig.subplots()

    ax.imshow(background, cmap='gray')

    # Draw all agents
    for agent in model.agents:
        if isinstance(agent, VehicleAgent):
            if hasattr(agent, 'active') and not agent.active:
                continue

            ax.add_patch(Circle(agent.pos, 3, facecolor=VEC_STATION_COLORS[agent.station.unique_id], edgecolor='black'))
            ax.text(agent.pos[0], agent.pos[1], str(agent.unique_id), ha='center', va='center')
        elif isinstance(agent, VECStationAgent):
            color = VEC_STATION_COLORS[agent.unique_id]
            ax.add_patch(Rectangle((agent.pos[0] - 3, agent.pos[1] - 3), 6, 6, facecolor=color))
            range_circle = Circle(agent.pos, agent.range, color=color, fill=False, linestyle='--')
            ax.add_patch(range_circle)

    ax.set_xlim(0, model.width)
    ax.set_ylim(0, model.height)

    solara.FigureMatplotlib(fig)


def render_model_orientations(model: VECModel):
    """
    Render the model with arrows indicating the orientation of the vehicles.
    """

    fig = Figure()
    ax = fig.subplots()

    for agent in model.agents:
        if isinstance(agent, VehicleAgent):
            if hasattr(agent, 'active') and not agent.active:
                continue

            arrow_length = 10
            angle_rad = math.radians(agent.angle)

            dx = arrow_length * math.cos(angle_rad)
            dy = arrow_length * math.sin(angle_rad)

            arrow = FancyArrow(agent.pos[0] - dx / 2, agent.pos[1] - dy / 2, dx, dy, head_width=5, head_length=6,
                               facecolor=VEC_STATION_COLORS[agent.station.unique_id], linewidth=0)
            ax.add_patch(arrow)

        elif isinstance(agent, VECStationAgent):
            color = VEC_STATION_COLORS[agent.unique_id]
            ax.add_patch(Rectangle((agent.pos[0] - 3, agent.pos[1] - 3), 6, 6, facecolor=color))
            range_circle = Circle(agent.pos, agent.range, color=color, fill=False, linestyle='--')
            ax.add_patch(range_circle)

    ax.set_xlim(0, model.width)
    ax.set_ylim(0, model.height)
    ax.set_aspect('equal')

    solara.FigureMatplotlib(fig)


def render_distance_chart(model: VECModel):
    """
    Render a chart showing the distances of vehicles from VEC stations.
    """

    fig = Figure()
    ax = fig.subplots()

    data = model.datacollector.get_agent_vars_dataframe()['Distances']
    filtered_distances = data.loc[data.index.get_level_values('AgentID') >= 10000]
    df = filtered_distances.unstack(level="AgentID")

    stations = {a.unique_id: VEC_STATION_COLORS[a.unique_id] for a in
                model.schedule.get_agents_by_type(VECStationAgent)}
    for station_id, color in stations.items():
        df[station_id].plot(ax=ax, color=color)

    ax.set_title('Distances from VEC stations')
    ax.set_xlabel('Step')
    ax.set_ylabel('Distance')
    solara.FigureMatplotlib(fig)


def make_render_station_vehicle_count_chart(tail=0):
    """
    Create a render function that renders a chart showing the vehicle count at VEC stations.
    """

    def render_station_vehicle_count_chart(model: VECModel):
        fig = Figure()
        ax = fig.subplots()

        data = model.datacollector.get_agent_vars_dataframe()['StationVehicleCount']
        filtered_counts = data.loc[data.index.get_level_values('AgentID') >= 10000]
        df = filtered_counts.unstack(level="AgentID")
        if tail > 0:
            df = df.tail(tail)

        stations = {a.unique_id: VEC_STATION_COLORS[a.unique_id] for a in
                    model.schedule.get_agents_by_type(VECStationAgent)}
        for station_id, color in stations.items():
            df[station_id].plot(ax=ax, color=color)

        ax.set_title('Vehicle count at VEC stations')
        ax.set_xlabel('Step')
        ax.set_ylabel('Vehicle count')
        solara.FigureMatplotlib(fig)

    return render_station_vehicle_count_chart


def make_render_station_load_chart(tail=0):
    """
    Create a render function that renders a chart showing the utilization of VEC stations.
    """

    def render_station_load_chart(model: VECModel):
        fig = Figure()
        ax = fig.subplots()

        data = model.datacollector.get_agent_vars_dataframe()['StationVehicleLoad']
        filtered_counts = data.loc[data.index.get_level_values('AgentID') >= 10000]
        df = filtered_counts.unstack(level="AgentID")
        if tail > 0:
            df = df.tail(tail)

        stations = {a.unique_id: VEC_STATION_COLORS[a.unique_id] for a in
                    model.schedule.get_agents_by_type(VECStationAgent)}
        for station_id, color in stations.items():
            df[station_id].plot(ax=ax, color=color)

        ax.axhline(y=1, color='gray', linestyle='--')

        ax.set_title('Utilization of VEC stations')
        ax.set_xlabel('Step')
        ax.set_ylabel('Utilization %')
        solara.FigureMatplotlib(fig)

    return render_station_load_chart


def render_vehicle_loads(model: VECModel):
    """
    Render a bar chart showing the loads of vehicles.
    """

    fig = Figure()
    ax = fig.subplots()

    data = model.datacollector.get_agent_vars_dataframe()['VehicleLoad']
    filtered_loads = data.loc[data.index.get_level_values('AgentID') < 10000]

    last_step_loads = filtered_loads.unstack(level="AgentID").tail(1).T
    last_step_loads = last_step_loads.dropna()
    vehicle_dict = {agent.unique_id: agent for agent in model.schedule.get_agents_by_type(VehicleAgent)}
    colors = [VEC_STATION_COLORS[vehicle_dict[vehicle_id].station.unique_id] for vehicle_id in last_step_loads.index]
    ax.bar(last_step_loads.index, last_step_loads.values.flatten(), color=colors)

    ax.set_title('Vehicle loads')
    ax.set_xlabel('Vehicle ID')
    ax.set_ylabel('Load')
    solara.FigureMatplotlib(fig)


# Colors for VEC stations
VEC_STATION_COLORS = {
    10001: "blue",
    10002: "orange",
    10003: "green",
    10004: "red",
    10005: "olive",
    10006: "pink",
    10007: "purple",
    10008: "brown",
    10009: "cyan"
}
