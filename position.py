""" This file adds classes for the coordinate system """

from abc import ABC, abstractmethod
import math
import random

class CoordinateSystemPoint(ABC):
    """ Abstract coordinate system point """

    @abstractmethod
    def __add__(self, other):
        """ Outputs the addition of itself with another point """

    @abstractmethod
    def __sub__(self, other):
        """ Outputs the subtraction of another point to itself (i.e. self - other) """

    @abstractmethod
    def __mul__(self, scalar):
        """ Outputs the multiplication by a scalar """

    @abstractmethod
    def __rmul__(self, scalar):
        """ Should call __mul__. Allows scalar * point operation to work """

    @abstractmethod
    def __truediv__(self, scalar):
        """ Outputs the division by a scalar """

    @abstractmethod
    def norm(self):
        """ Returns the norm of the point """

    @abstractmethod
    def random_unit_vector(self):
        """ Returns a random unit length vector """

    @abstractmethod
    def __repr__(self):
        """ Return the point representation"""


class Euclidean2D(CoordinateSystemPoint):
    """ Implements the Euclidean 2D coordinate system """

    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y

    def __add__(self, other):
        if isinstance(other, Euclidean2D):
            return Euclidean2D(self.x + other.x, self.y + other.y)
        return NotImplemented

    def __sub__(self, other):
        if isinstance(other, Euclidean2D):
            return Euclidean2D(self.x - other.x, self.y - other.y)
        return NotImplemented

    def __mul__(self, scalar):
        if isinstance(scalar, (int, float)):
            return Euclidean2D(self.x * scalar, self.y * scalar)
        return NotImplemented

    def __rmul__(self, scalar):
        return self.__mul__(scalar)

    def __truediv__(self, scalar):
        if isinstance(scalar, (int, float)):
            return Euclidean2D(self.x / scalar, self.y / scalar)
        return NotImplemented

    def norm(self):
        return math.sqrt(self.x * self.x + self.y * self.y)

    def random_unit_vector(self):
        x = random.random()
        y = random.random()
        random_point = Euclidean2D(x, y)
        return random_point / random_point.norm()

    def __repr__(self):
        return f"E2({self.x}, {self.y})"
