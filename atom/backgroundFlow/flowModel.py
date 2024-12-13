from dataclasses import dataclass
import pandas as pd
import numpy as np


@dataclass(kw_only=True)
class FlowModel:
    """
    The background flow object describes the background atmospheric velocity and temperature fields around which spatial fluctuations are to be calculated.
    """


@dataclass(kw_only=True)
class homogeneous(FlowModel):
    """
    Constant homogeneous background flow. Spatial average values of u, v, and T should be supplied from the AuxData (sonic anemometer). If no average values for u, v, or T are supplied, defaults to None.
    """

    def __post_init__(self, meanU, meanV, meanT):
        self.u = meanU
        self.v = meanV
        self.T = meanT

    def rotateVel(self, flowDir: float = 0.0) -> None:
        """
        Rotate velocity field to match measured flow direction.
        """
        rotationMatrix = np.array(
            [np.cos(flowDir), -np.sin(flowDir)], [np.sin(flowDir), np.cos(flowDir)]
        )

        self.u, self.v = rotationMatrix * np.array([self.u, self.v])


@dataclass(kw_only=True)
class logLaw(FlowModel):
    """
    The log wind profile is a semi-empirical relationship commonly used to describe the vertical distribution of horizontal mean wind speeds within the lowest portion of the planetary boundary layer.
    reference: https://en.wikipedia.org/wiki/Log_wind_profile
    """

    frictionVel: float
    kappa: float
    roughnessHeight: float
    displacementHeight: float

    def __post_init__(self):
        self.u = self.logLawProfile()
        self.v = 0.0

    def logLawProfile(self, zVec):
        return (
            self.frictionVel
            / self.kappa
            * (np.log((zVec - self.displacementHeight) / self.roughnessHeight))
        )


@dataclass(kw_only=True)
class powerLaw(FlowModel):
    """
    Calculates velocity following a power law with a reference velocity at a reference height and a shear exponent.
    reference: https://en.wikipedia.org/wiki/Wind_profile_power_law
    """

    uRef: float
    zRef: float
    shearExponent: float

    def __post_init__(self):
        self.u = self.powerLawProfile()
        self.v = 0.0

    def powerLawProfile(self, zVec):
        return self.uRef * (zVec / self.zRef) ** self.shearExponent
