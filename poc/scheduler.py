from typing import Iterable, List, TypeVar, Type

import mesa
from mesa import Model, Agent

T = TypeVar('T', bound=Agent)


class RandomActivationBySortedType(mesa.time.RandomActivationByType):
    def __init__(self, model: Model, types: List[type(Agent)], agents: Iterable[Agent] | None = None) -> None:
        super().__init__(model, agents)
        self.types = types

    def step(self, shuffle_types: bool = False, shuffle_agents: bool = True) -> None:
        if shuffle_types:
            self.model.random.shuffle(self.types)
        for agent_type in self.types:
            if agent_type not in self._agents_by_type:
                continue
            self.step_type(agent_type, shuffle_agents)

        self.steps += 1
        self.time += 1

    def get_agents_by_type(self, agent_type: Type[T]) -> List[T]:
        return self._agents_by_type.get(agent_type, [])
