""" Basic Types """


from dataclasses import dataclass


class NodeID(int):
    """NodeID"""

@dataclass
class Ping:
    """Ping"""
    base: float
    std_dev: float

@dataclass
class Event:
    """Network event"""

    source: NodeID
    target: NodeID
    timestamp: float
    id: int

    def __repr__(self):
        return f"Event(source={self.source}, target={self.target}, timestamp={self.timestamp})"
