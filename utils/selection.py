from typing import List
import numpy as np
import random


class Roulette(object):

    class Element(object):
        def __init__(self, name: str, selected: int = 0):
            self.name = name
            self.selected = selected

        def record(self):
            self.selected += 1

        @property
        def score(self):
            return 1.0 / (self.selected + 1)
    
    def __init__(self, layer_types: List[str], layer_conditions: dict, use_heuristic: bool = True):
        self.__pool = {name: self.Element(name=name) for name in layer_types}
        self.__layer_conditions = layer_conditions
        self.__use_heuristic = use_heuristic