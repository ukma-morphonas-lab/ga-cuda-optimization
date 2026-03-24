
from abc import ABC, abstractmethod
from typing import Any


class FitnessMetric(ABC):
    @abstractmethod
    def get_name(self) -> str:
        pass

    @abstractmethod
    def extract_value(self, result: Any) -> float:
        pass

    @abstractmethod
    def is_better(self, value1: float, value2: float) -> bool:
        pass

    @abstractmethod
    def get_goal_description(self) -> str:
        pass


class MaximizationMetric(FitnessMetric):
    def __init__(self, name: str = "fitness", attribute: str = "final_fitness"):
        self._name = name
        self._attribute = attribute

    def get_name(self) -> str:
        return self._name

    def extract_value(self, result: Any) -> float:
        return float(getattr(result, self._attribute))

    def is_better(self, value1: float, value2: float) -> bool:
        return value1 > value2

    def get_goal_description(self) -> str:
        return "higher"


class MinimizationMetric(FitnessMetric):
    def __init__(self, name: str = "cost", attribute: str = "final_path_cost"):
        self._name = name
        self._attribute = attribute

    def get_name(self) -> str:
        return self._name

    def extract_value(self, result: Any) -> float:
        return float(getattr(result, self._attribute))

    def is_better(self, value1: float, value2: float) -> bool:
        return value1 < value2

    def get_goal_description(self) -> str:
        return "lower"


def fitness_maximization(name: str = "fitness") -> FitnessMetric:
    return MaximizationMetric(name, "final_fitness")


def cost_minimization(name: str = "path cost") -> FitnessMetric:
    return MinimizationMetric(name, "final_path_cost")
