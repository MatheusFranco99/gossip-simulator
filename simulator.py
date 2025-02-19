""" Simulator """

import collections
from queue import PriorityQueue
from dataclasses import dataclass
import random
from basic_types import Event, NodeID
from clustering import create_cluster_nodes
from gossip_algorithm import GossipAlgorithm
from metrics import Metric, Metrics
from network import Network
from node import Node
import numpy as np
import matplotlib.pyplot as plt

from position import CoordinateSystemPoint, Euclidean2D
from attacker import Attacker, LowestTimeEstimator

class Simulator:
    """Simulator"""

    def __init__(
        self,
        network: Network,
        gossip_algorithm: GossipAlgorithm,
    ):
        # Parameters
        self.dimension = 2
        self.gossip_algorithm = gossip_algorithm
        self.network = network
        self.first_source: Node | None = None

    def setup(self) -> None:
        """Setups the simulator for execution"""
        # Select a source
        self.first_source: Node = self.network.nodes[random.randint(0, len(self.network.nodes) - 1)]

    def select_targets(self, node_id: NodeID) -> list[NodeID]:
        """Selects a random target for node_id to send a message"""
        return self.gossip_algorithm.select_targets(node_id)

    def run(
        self,
        use_max_time: bool = False,
        max_time: float = 0,
        stop_when_all_informed: bool = True,
        attackers: list[Attacker] = None,
        msg_receival_limit: int = 10,
    ) -> tuple[Metric, Metric]:
        """Executes the simulation by:
        - Choosing a random source
        - Iterating over the network events until max time or all are informed
        """


        current_time: float = 0
        current_id: int = 0

        # Create queue
        queue: PriorityQueue = PriorityQueue()


        # Counter of the number of times a node received a message
        node_receipt_counter: dict[NodeID, int] = collections.defaultdict(int)

        def add_event(queue: PriorityQueue, event: Event) -> PriorityQueue:
            queue.put((event.timestamp, event.id, event))
            return queue

        # Create initial event
        targets = self.select_targets(self.first_source.node_id)
        node_receipt_counter[self.first_source.node_id] += 1
        for target in targets:
            initial_event = Event(
                source=self.first_source.node_id,
                target=target,
                timestamp=current_time
                + self.network.get_delay(self.first_source.node_id, target),
                id=current_id,
            )
            current_id += 1
            add_event(queue, initial_event)

        # Metrics
        arrival_time: dict[NodeID, float] = (
            {}
        )  # NodeID and the time it first received the message
        arrival_time[self.first_source.node_id] = 0

        # Iterate
        while not queue.empty():
            if use_max_time and current_time > max_time:
                break
            if stop_when_all_informed and len(arrival_time) == len(self.network.nodes):
                break

            # Get next event
            event: Event | None = None
            event_time, _, event = queue.get()

            # Check if node is still processing events
            if node_receipt_counter[event.target] > msg_receival_limit:
                continue
            node_receipt_counter[event.target] += 1

            # print("Processing event:", event)

            # Send event to attackers
            for attacker in attackers:
                if attacker.has_access_to_event(event):
                    attacker.process_event(event)

            # Add target to active, if not yet active
            if event.target not in arrival_time:
                print(len(arrival_time)/len(self.network.nodes), queue.qsize())
                arrival_time[event.target] = event.timestamp

            # Process message (Cobra-walk algorithm)
            new_source = event.target
            targets = self.select_targets(new_source)
            for target in targets:
                queue = add_event(
                    queue,
                    Event(
                        source=new_source,
                        target=target,
                        timestamp=event.timestamp
                        + self.network.get_delay(new_source, target),
                        id=current_id,
                    ),
                )
                current_id += 1

            # Update current time
            current_time = event_time

        # Runs attackers
        attacker_results = []
        # for attacker in attackers:
        #     guess = attacker.guess()
        #     attacker_results.append(guess == self.first_source.node_id)

        # Compute stretch
        metrics = Metrics(self.network, self.first_source, arrival_time)
        stretch = metrics.get_stretch()

        return stretch, Metric(attacker_results)
