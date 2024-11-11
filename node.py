""" Node """

from dataclasses import dataclass

from basic_types import NodeID
from position import CoordinateSystemPoint


@dataclass
class Node:
    """Node"""

    node_id: NodeID
    pos: CoordinateSystemPoint
