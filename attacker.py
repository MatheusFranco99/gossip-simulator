""" This file adds the Attacker class (and some examples) for the anonymity simulation """
from abc import ABC, abstractmethod

from basic_types import Event, NodeID

class Attacker(ABC):
    """ This abstract class represents a general anonymity attacker """
    name: str = "Abstract"

    def __init__(self, node_ids: list[NodeID], curious_node_ids: list[NodeID]):
        self.node_ids = node_ids
        self.curious_node_ids = curious_node_ids
        self.num_honest_peers = len(node_ids) - len(curious_node_ids)

        # Initialize probabilities as 0
        self.probability: dict[NodeID, float] = {}
        for peer_id in self.node_ids:
            if peer_id not in self.curious_node_ids:
                self.probability[peer_id] = 0

    @abstractmethod
    def process_event(self, event: Event) -> None:
        """ Internal method. Process an event and update the probabilities for each possible creator """

    @abstractmethod
    def guess(self) -> NodeID:
        """ This function returns the attacker's NodeID guess for the creator of the message """

    @abstractmethod
    def process_all_events(self, curious_nodes_events: dict[NodeID, list[Event]]) -> None:
        """ Receives events per curious node and process all events to update its probability map """

class UniformEstimator(Attacker):
    """ The UniformEstimator is an attacker that inits with equal probability for all known NodeIDs.
    Given an event, to update the probability of the sender it computes the factor:
    f = max(0.1, delivery_time)/0.1
    and increment the sender probability by:
    p = p + 1/N * 1/f
    Then it normalizes the probabilities.
    Notice that later the message, greater f, and smaller 1/f.
    Eariler the message, smaller the f, and greater 1/f.
    """
    name: str = "Uniform estimator"

    def __init__(self, node_ids: list[NodeID], curious_node_ids: list[NodeID]):
        super().__init__(node_ids, curious_node_ids)

        uniform_probability = 1/self.num_honest_peers
        for peer_id in self.node_ids:
            if peer_id not in self.curious_node_ids:
                self.probability[peer_id] = uniform_probability

    def normalize(self) -> None:
        """ This function normalizes all probabilities """
        total_sum = sum(self.probability.values())
        for i, p in self.probability.items():
            self.probability[i] = p/total_sum

    def process_event(self, event: Event) -> None:
        # If sender is a curious node, just return
        if event.target in self.curious_node_ids:
            return

        delivery_time = max(0.1, event.timestamp)
        delivery_time_factor = delivery_time/0.1
        self.probability[event.source] = self.probability[event.target] +  1/(self.num_honest_peers) * (1/delivery_time_factor)

        self.normalize()

    def guess(self) -> NodeID:
        highest_probability = 0
        highest_probability_peer_id = 0
        for i, p in self.probability.items():
            if p > highest_probability:
                highest_probability = p
                highest_probability_peer_id = i
        return highest_probability_peer_id

    def process_all_events(self, curious_nodes_events: dict[NodeID, list[Event]]) -> None:
        for events in curious_nodes_events.values():
            for event in events:
                self.process_event(event)


class LowestTimeEstimator(Attacker):
    """ The LowestTimeEstimator is an attacker that
    guesses the source according to the message with lowest time. """
    name: str = "Lowest time estimator"

    def __init__(self, node_ids: list[NodeID], curious_node_ids: list[NodeID]):
        super().__init__(node_ids, curious_node_ids)

        # Starts with equal probability
        uniform_probability = 1/self.num_honest_peers
        for peer_id in self.node_ids:
            if peer_id not in self.curious_node_ids:
                self.probability[peer_id] = uniform_probability

        self.lowest_time = float("inf")

    def set_probabilities_to_zero(self) -> None:
        """ This function set all probabilities to zero """
        for i in self.probability:
            self.probability[i] = 0

    def process_event(self, event: Event) -> None:
        # If sender is a curious node, just return
        if event.target in self.curious_node_ids:
            return

        if event.timestamp < self.lowest_time:
            self.lowest_time = event.timestamp
            self.set_probabilities_to_zero()
            self.probability[event.target] = 1

    def guess(self) -> NodeID:
        highest_probability = 0
        highest_probability_peer_id = 0
        for i, p in self.probability.items():
            if p > highest_probability:
                highest_probability = p
                highest_probability_peer_id = i
        return highest_probability_peer_id

    def process_all_events(self, curious_nodes_events: dict[NodeID, list[Event]]) -> None:
        for events in curious_nodes_events.values():
            for event in events:
                self.process_event(event)
