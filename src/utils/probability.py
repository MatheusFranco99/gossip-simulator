""" Probability """

import copy
import random


def bernoulli_event(probability: float) -> bool:
    """Sample an event"""
    # return random.choices([True, False], weights=[probability, (1-probability)])[0]
    p = random.random()
    return p <= probability


def select_from_group(population: list):
    """Sample an event"""
    return random.choice(population)


def select_samples_from_group_with_replacement(population: list, k: int = 1) -> list:
    """Select samples from a population with replacement"""
    return random.choices(population, k=k)


def select_samples_from_group_without_replacement(population: list, k: int = 1) -> list:
    """Select samples from a population without replacement"""
    copied_population = copy.copy(population)
    random.shuffle(copied_population)
    return copied_population[:k]


def hypergeometric_sample(a: int, b: int, k: int) -> int:
    """Get a sample from the hypergeometric distribution"""
    space = [1] * a + [0] * b
    random.shuffle(space)
    selection_event = space[:k]
    sum_of_events_a = sum(selection_event)
    return sum_of_events_a
