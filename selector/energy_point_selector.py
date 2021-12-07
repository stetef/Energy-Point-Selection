"""Energy point selector."""

import numpy as np
import sklearn


class Selector:
    """Select the features that most describe the reference standards."""

    def __init__(self, references):
        """Init function."""
        self.References = references

    def select_energy_points(self, n_points=None):
        """Return the n_points that are most important."""
        if n_points is None:
            n_points = len(self.References)
        return n_points
