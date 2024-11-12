""" Simulator """

from queue import PriorityQueue
from dataclasses import dataclass
import random

from matplotlib import animation
from basic_types import NodeID
from network import Network
from node import Node
import numpy as np
import matplotlib.pyplot as plt

from position import CoordinateSystemPoint, Euclidean2D
from spatial_gossip import get_spatial_gossip_probability_vector

@dataclass
class Event:
    """ Network event """
    source: NodeID
    target: NodeID
    timestamp: float

    def __repr__(self):
        return f"Event(source={self.source}, target={self.target}, timestamp={self.timestamp})"

class Simulator:
    """ Simulator """

    def __init__(self, num_nodes: int, rho: float, cobra_walk_rho: float, grid_size: int = 100):
        # Parameters
        self.dimension = 2
        self.rho = rho
        self.cobra_walk_rho = cobra_walk_rho
        self.grid_size = grid_size

        # Build network
        self.nodes: list[Node] = []
        for i in range(num_nodes):
            x = random.randint(0, grid_size)
            y = random.randint(0, grid_size)
            node = Node(i, pos = Euclidean2D(x, y))
            self.nodes.append(node)

        self.network: Network = Network(self.nodes)

        # Spatial gossip structure
        self.spatial_gossip_vectors: dict[NodeID, dict[NodeID, float]] = {}

    def setup(self) -> None:
        """ Setups the simulator for execution """
        for node in self.nodes:
            self.spatial_gossip_vectors[node.node_id] = get_spatial_gossip_probability_vector(self.network, node, dimension=self.dimension, rho=self.rho)

            # Uniform distribution
            # self.spatial_gossip_vectors[node.node_id] = {}
            # for other_node in self.nodes:
            #     if other_node.node_id == node.node_id:
            #         continue
            #     self.spatial_gossip_vectors[node.node_id][other_node.node_id] = 1/(len(self.nodes)-1)

    def select_random_target(self, node_id: NodeID) -> NodeID:
        """ Selects a random target for node_id to send a message """
        return np.random.choice(list(self.spatial_gossip_vectors[node_id].keys()), p = list(self.spatial_gossip_vectors[node_id].values()))

    def get_base_delay(self, node_1: NodeID, node_2: NodeID) -> float:
        """ Returns the latency between two nodes """
        distance_vector: CoordinateSystemPoint = (self.nodes[node_1].pos - self.nodes[node_2].pos)
        return distance_vector.norm()

    def get_delay(self, node_1: NodeID, node_2: NodeID) -> float:
        """ Returns the latency between two nodes """
        distance_vector: CoordinateSystemPoint = (self.nodes[node_1].pos - self.nodes[node_2].pos)
        return distance_vector.norm() * (1 + random.random()/10)

    def show_network(self) -> None:
        """ Plots the 2D network in a grid """
        x = []
        y = []
        for node in self.nodes:
            pos: Euclidean2D = node.pos
            x.append(pos.x)
            y.append(pos.y)

        plt.figure(figsize = (10,8))
        plt.scatter(x, y)
        plt.grid()
        plt.show()


    def run(self, use_max_time: bool = False, max_time: float = 0, stop_until_all_informed: bool = True) -> list[float]:
        """ Executes the simulation by:
        - Choosing a random source
        - Iterating over the network events unitl max time or all are informed
        """

        current_time: float = 0
        # Select a source
        source = self.nodes[random.randint(0, len(self.nodes) - 1)]
        # Create queue
        queue: PriorityQueue = PriorityQueue()
        arrival_time: dict[NodeID, float] = {} # NodeID and the time it first received the message
        arrival_time[source.node_id] = 0

        def add_event(queue: PriorityQueue, event: Event) -> PriorityQueue:
            # print("QUEUE")
            # for elm in queue.queue:
            #     print("\t",elm)
            # print("\tNew event:", event)
            # print()
            queue.put((event.timestamp, event))
            return queue

        # Create initial event
        target = self.select_random_target(source.node_id)
        initial_event = Event(source = source.node_id, target = target, timestamp=current_time + self.get_delay(source.node_id, target))
        add_event(queue, initial_event)


        fig, ax = plt.subplots()
        ax.set_xlim(0, self.grid_size)  # Set according to your grid size
        ax.set_ylim(0, self.grid_size)
        ax.set_title('Network Simulation')

        scatter = ax.scatter([node.pos.x for node in self.nodes], [node.pos.y for node in self.nodes], s=100)
        node_texts = {}
        for node in self.nodes:
            node_text = ax.text(node.pos.x, node.pos.y, str(node.node_id), ha='center', va='center')
            node_texts[node.node_id] = node_text
        arrow = None
        next_event_arrow = None
        next_event_arrow2 = None
        event_text = ax.text(0.05, 0.95, '', transform=ax.transAxes, ha='left')

        def init():
            scatter.set_facecolor(['blue'] * len(self.nodes))
            for text in node_texts.values():
                text.set_color('white')
            return scatter, event_text

        def update(frame):
            nonlocal current_time, queue, arrow, next_event_arrow, next_event_arrow2

            if queue.empty() or (use_max_time and current_time > max_time) or (stop_until_all_informed and len(arrival_time) == len(self.nodes)):
                anim.event_source.stop()
                return scatter, event_text

            event_time, event = queue.get()
            current_time = event_time
            source_node = self.nodes[event.source]
            target_node = self.nodes[event.target]

            if arrow:
                arrow.remove()
            arrow = ax.arrow(source_node.pos.x, source_node.pos.y, target_node.pos.x - source_node.pos.x, target_node.pos.y - source_node.pos.y, head_width=1, color='gray')

            if event.target not in arrival_time:
                arrival_time[event.target] = event.timestamp
                colors = scatter.get_facecolor()
                colors[target_node.node_id] = [0,1,0,1]  # Mark informed nodes in green
                colors[source_node.node_id] = [0,1,0,1]  # Mark informed nodes in green
                scatter.set_facecolor(colors)
                node_texts[event.target].set_color('green')

            # for node_id, node_text in node_texts.items():
            #     node_text.set_text(str(node_id))

            new_source = event.target
            target = self.select_random_target(new_source)
            queue = add_event(queue, Event(source=new_source, target=target, timestamp=event.timestamp + self.get_delay(new_source, target)))

            # if next_event_arrow is not None:
            #     next_event_arrow.remove()
            #     next_event_arrow = None
            # if next_event_arrow2 is not None:
            #     next_event_arrow2.remove()
            #     next_event_arrow2 = None
            # new_source_node = self.nodes[new_source]
            # new_target_node = self.nodes[target]
            # next_event_arrow = ax.arrow(new_source_node.pos.x, new_source_node.pos.y, new_target_node.pos.x - new_source_node.pos.x, new_target_node.pos.y - new_source_node.pos.y, head_width=1, color='purple', linestyle="--")

            random_value = random.random()
            if random_value <= self.cobra_walk_rho:
                target2 = self.select_random_target(new_source)
                if target2 != target:
                    queue = add_event(queue, Event(source=new_source, target=target2, timestamp=event.timestamp + self.get_delay(new_source, target2)))

                    # new_source_node = self.nodes[new_source]
                    # new_target_node = self.nodes[target2]
                    # next_event_arrow2 = ax.arrow(new_source_node.pos.x, new_source_node.pos.y, new_target_node.pos.x - new_source_node.pos.x, new_target_node.pos.y - new_source_node.pos.y, head_width=1, color='purple', linestyle="--")


            event_text.set_text(f"Current Event: {event.source} -> {event.target} at {event.timestamp:.2f}")
            return scatter, event_text, arrow

        anim = animation.FuncAnimation(fig, update, init_func=init, frames = 500, blit=True, interval=1000)
        anim.save('gossip.mp4', writer='ffmpeg', fps=30)
        # plt.show()

        stretch = []
        for node, time in arrival_time.items():
            if node == source.node_id:
                continue
            stretch.append(time / self.get_base_delay(source.node_id, node))

        return stretch

        # # Iterate
        # while (not queue.empty()):
        #     if use_max_time and current_time > max_time:
        #         break
        #     if stop_until_all_informed and len(arrival_time) == len(self.nodes):
        #         break

        #     # Get next event
        #     event: Event = None
        #     event_time, event = queue.get()
        #     # print("Processing event:", event)

        #     # Add target to active, if not yet active
        #     if event.target not in arrival_time:
        #         arrival_time[event.target] = event.timestamp

        #     # Process message (Cobra-walk algorithm)
        #     new_source = event.target
        #     target = self.select_random_target(new_source)
        #     queue = add_event(queue, Event(source = new_source, target = target, timestamp = event.timestamp + self.get_delay(new_source, target)))

        #     random_value = random.random()
        #     cobra_partition = (random_value <= self.cobra_walk_rho)
        #     if cobra_partition:
        #         target2 = self.select_random_target(new_source)
        #         if target2 != target:
        #             queue = add_event(queue, Event(source = new_source, target = target2, timestamp = event.timestamp + self.get_delay(new_source, target2)))

        #     # Update current time
        #     current_time = event_time

        # # Compute stretch
        # stretch: list[float] = []
        # for node, time in arrival_time.items():
        #     if node == source.node_id:
        #         continue
        #     stretch.append(time / self.get_base_delay(source.node_id, node))

        # return stretch
