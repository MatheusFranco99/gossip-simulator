""" Simulator """

from queue import PriorityQueue
from dataclasses import dataclass
import random
from basic_types import Event, NodeID
from network import Network
from node import Node
import numpy as np
import matplotlib.pyplot as plt

from position import CoordinateSystemPoint, Euclidean2D
from spatial_gossip import get_spatial_gossip_probability_vector
from attacker import LowestTimeEstimator

class Simulator:
    """Simulator"""

    def __init__(
        self,
        network: Network,
        rho: float,
        cobra_walk_rho: float,
        spatial_gossip: bool = True,
        f: float = 0.1,
        num_attackers: int = 5,
    ):
        # Parameters
        self.dimension = 2
        self.rho = rho
        self.cobra_walk_rho = cobra_walk_rho
        self.spatial_gossip = spatial_gossip
        self.network = network
        self.f = f
        self.num_attackers = num_attackers

        # Spatial gossip structure
        self.spatial_gossip_vectors: dict[NodeID, dict[NodeID, float]] = {}

    def setup(self) -> None:
        """Setups the simulator for execution"""
        for node in self.network.nodes:
            # Spatial gossip
            if self.spatial_gossip:
                self.spatial_gossip_vectors[node.node_id] = (
                    get_spatial_gossip_probability_vector(
                        self.network, node, dimension=self.dimension, rho=self.rho
                    )
                )

            # Uniform distribution
            else:
                self.spatial_gossip_vectors[node.node_id] = {}
                for other_node in self.network.nodes:
                    if other_node.node_id == node.node_id:
                        continue
                    self.spatial_gossip_vectors[node.node_id][other_node.node_id] = (
                        1 / (len(self.network.nodes) - 1)
                    )

    def select_random_target(self, node_id: NodeID) -> NodeID:
        """Selects a random target for node_id to send a message"""
        return np.random.choice(
            list(self.spatial_gossip_vectors[node_id].keys()),
            p=list(self.spatial_gossip_vectors[node_id].values()),
        )

    def run(
        self,
        use_max_time: bool = False,
        max_time: float = 0,
        stop_when_all_informed: bool = True,
    ) -> list[float]:
        """Executes the simulation by:
        - Choosing a random source
        - Iterating over the network events unitl max time or all are informed
        """

        all_node_ids: list[NodeID] = [node.node_id for node in self.network.nodes]

        # Select a source
        source = self.network.nodes[random.randint(0, len(self.network.nodes) - 1)]
        first_source = source

        # Create list of sets of curious nodes
        num_curious_nodes = int(self.f * len(self.network.nodes))
        curious_sets = []
        all_node_ids_except_source: list[NodeID] = [node.node_id for node in self.network.nodes if node.node_id != first_source.node_id]
        for _ in range(self.num_attackers):
            curious_sets.append(set(random.sample(all_node_ids_except_source, num_curious_nodes)))

        # For each possible curious node, its list of events
        curious_events = {node_id: [] for curious_set in curious_sets for node_id in curious_set}

        current_time: float = 0

        # Create queue
        queue: PriorityQueue = PriorityQueue()

        def add_event(queue: PriorityQueue, event: Event) -> PriorityQueue:
            queue.put((event.timestamp, event))
            return queue

        # Create initial event
        target = self.select_random_target(source.node_id)
        initial_event = Event(
            source=source.node_id,
            target=target,
            timestamp=current_time + self.network.get_delay(source.node_id, target),
        )
        add_event(queue, initial_event)

        # Metrics
        arrival_time: dict[NodeID, float] = (
            {}
        )  # NodeID and the time it first received the message
        arrival_time[source.node_id] = 0

        # Iterate
        while not queue.empty():
            if use_max_time and current_time > max_time:
                break
            if stop_when_all_informed and len(arrival_time) == len(self.network.nodes):
                break

            # Get next event
            event: Event = None
            event_time, event = queue.get()
            # print("Processing event:", event)

            # If target is curious, add event to curious events
            if event.target in curious_events:
                curious_events[event.target].append(event)

            # Add target to active, if not yet active
            if event.target not in arrival_time:
                # print(len(arrival_time)/len(self.network.nodes), queue.qsize())
                arrival_time[event.target] = event.timestamp

            # Process message (Cobra-walk algorithm)
            new_source = event.target
            target = self.select_random_target(new_source)
            queue = add_event(
                queue,
                Event(
                    source=new_source,
                    target=target,
                    timestamp=event.timestamp + self.network.get_delay(new_source, target),
                ),
            )

            random_value = random.random()
            cobra_partition = random_value <= self.cobra_walk_rho
            if cobra_partition:
                target2 = self.select_random_target(new_source)
                if target2 != target:
                    queue = add_event(
                        queue,
                        Event(
                            source=new_source,
                            target=target2,
                            timestamp=event.timestamp
                            + self.network.get_delay(new_source, target2),
                        ),
                    )

            # Update current time
            current_time = event_time

        # Runs attackers
        attacker_results = []
        for curious_set in curious_sets:
            # Create attacker
            attacker = LowestTimeEstimator(node_ids = all_node_ids, curious_node_ids = list(curious_set))
            # Load attacker with events
            events_for_attacker: dict[NodeID, list[Event]] = {}
            for node_id, events in curious_events.items():
                if node_id in curious_set:
                    events_for_attacker[node_id] = events
            attacker.process_all_events(events_for_attacker)
            # Run attacker's guess
            guess = attacker.guess()
            attacker_results.append(guess == first_source.node_id)


        # Compute stretch
        stretch: list[float] = []
        for node, time in arrival_time.items():
            if node == source.node_id:
                continue
            stretch.append(time / self.network.get_base_delay(source.node_id, node))

        return stretch, attacker_results
