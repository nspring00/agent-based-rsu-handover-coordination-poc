from typing import Iterable, List, TypeVar, Type

from mesa import Model, Agent
from mesa.time import RandomActivationByType

T = TypeVar('T', bound=Agent)


class RandomActivationBySortedType(RandomActivationByType):
    """
    A scheduler which activates each type of agent once per step, in the specified agent order, but all the
    same type are activated in a random order. This is equivalent to the RandomActivationByType model, but with a fixed
    order of agent types.
    """

    def __init__(self, model: Model, types: List[type(Agent)], agents: Iterable[Agent] | None = None) -> None:
        """
        Create a new RandomActivationBySortedType scheduler.

        Args:
            model (Model): The model to which the schedule belongs
            types (List[type(Agent)]): A list of agent types to be executed in the specified order
            agents (Iterable[Agent], None, optional): An iterable of agents who are controlled by the schedule
        """

        super().__init__(model, agents)
        self.types = types

    def step(self, shuffle_types: bool = False, shuffle_agents: bool = True) -> None:
        """
        Executes the step of each agent type, one at a time, in the specified order.

        Args:
            shuffle_types: If True, the order of execution of each types is
                           shuffled.
            shuffle_agents: If True, the order of execution of each agents in a
                            type group is shuffled.
        """

        if shuffle_types:
            self.model.random.shuffle(self.types)
        for agent_type in self.types:
            if agent_type not in self._agents_by_type:
                continue
            self.step_type(agent_type, shuffle_agents)

        self.steps += 1
        self.time += 1

    def get_agents_by_type(self, agent_type: Type[T]) -> List[T]:
        """
        Returns the agents of a given type.
        """

        return self._agents_by_type.get(agent_type, [])
