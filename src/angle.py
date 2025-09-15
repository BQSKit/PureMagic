"""Module for representing angles for Pauli Basis Measurements.
Copied from bftp/quilt on github
"""

from __future__ import annotations

from dataclasses import dataclass

from fractions import Fraction

from math import gcd

from numpy import ceil
from numpy import floor
from numpy import pi


@dataclass
class Angle:
    """
    Angles are represented exactly using integer denominators and numerators.
    """

    def __init__(self, numerator: int, denominator: int) -> None:
        """Initialize an Angle."""
        self.numerator = int(numerator)
        self.denominator = int(denominator)
        self.simplify()

    def simplify(self) -> None:
        """Simplify the Angle."""
        factor = gcd(self.numerator, self.denominator)
        self.numerator //= factor
        self.denominator //= factor

    def is_clifford(self) -> bool:
        """Returns True if this Angle is a Clifford rotation."""
        self.simplify()
        return self.denominator in (-4, -2, -1, 0, 1, 2, 4)

    @property
    def value(self) -> float:
        """The float value represented by this Angle."""
        return self.numerator * pi / self.denominator

    @staticmethod
    def from_float(value: float, decimals: int = 16) -> Angle:
        """
        Convert float value into a Angle object with given precision.

        Args:
            value (float): The float value to convert to a Angle.

            decimals (int): The precision to use when converting the
                denominator.
        """
        value = (value % (2 * pi)) / pi
        fraction = Fraction(value).limit_denominator(1000)
        return Angle(fraction.numerator, fraction.denominator)

    def __add__(self, other: Angle) -> Angle:
        x_n, x_d = self.numerator, self.denominator
        y_n, y_d = other.numerator, other.denominator
        new_n = x_n * y_d + y_n * x_d
        new_d = x_d * y_d
        return Angle(new_n, new_d)

    def __repr__(self) -> str:
        if self.denominator == 1:
            return f"Angle({self.numerator}pi)"
        elif self.numerator == 1:
            return f"Angle(pi/{self.denominator})"
        return f"Angle({self.numerator}pi/{self.denominator})"

    def __hash__(self) -> int:
        return hash((self.numerator, self.denominator))
