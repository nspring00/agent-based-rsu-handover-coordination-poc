from typing import Iterable, List

import mesa
from mesa import Model, Agent


class RandomActivationBySortedType(mesa.time.RandomActivationByType):
    def __init__(self, model: Model, types: List[type(Agent)], agents: Iterable[Agent] | None = None) -> None:
        super().__init__(model, agents)
        self.types = types

    def step(self, shuffle_types: bool = False, shuffle_agents: bool = True) -> None:
        self.model.random.shuffle(self.types)
        for agent_type in self.types:
            if agent_type not in self._agents_by_type:
                continue
            self.step_type(agent_type, shuffle_agents)

        self.steps += 1
        self.time += 1
