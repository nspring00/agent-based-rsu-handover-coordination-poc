import solara
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle, Circle

from poc.simple import VehicleAgent, VEC_STATION_COLORS, VECStationAgent, VECModel


def agent_portrayal(agent):
    # Render vehicle agent
    if isinstance(agent, VehicleAgent):
        if hasattr(agent, 'active') and not agent.active:
            return {
                "color": "black",
                "s": 50,
            }
        return {
            "color": VEC_STATION_COLORS[agent.station.unique_id],
            "s": 50,
        }

    # TODO check how to get size/shape working
    # Render VEC station agent
    if isinstance(agent, VECStationAgent):
        portrayal = {
            # "shape": "rect",
            "color": VEC_STATION_COLORS[agent.unique_id],
            "s": 100,
            # "filled": "true",
            # "layer": 0,
            # "w": 1,
            # "h": 1,
        }
        return portrayal

    assert False


def render_model_with_bg(background):
    return lambda model: render_model(model, background)


def render_model(model, background=None):
    fig = Figure()
    ax = fig.subplots()

    if background is None:

        min_x = min(waypoint[0] for waypoint in model.waypoints)
        max_x = max(waypoint[0] for waypoint in model.waypoints)
        min_y = min(waypoint[1] for waypoint in model.waypoints)
        max_y = max(waypoint[1] for waypoint in model.waypoints)
        road_width = max_x - min_x
        road_height = max_y - min_y

        # Draw the road
        road = Rectangle((min_x, min_y), road_width, road_height, linewidth=3, edgecolor='gray', facecolor='none')
        ax.add_patch(road)

    else:
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
            range_circle = Circle(agent.pos, agent.distance_threshold * agent.range, color=color, fill=False,
                                  linestyle='--',
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
