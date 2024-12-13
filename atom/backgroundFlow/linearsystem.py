from dataclasses import dataclass
import numpy as np
import xarray as xr
from atom import utils


@dataclass
class LinearSystem:
    """
    This class is responsible for assembling all the necessary information for the linear system solution as part of the AT process.

    .. math::

        Gf = b


    :math:`G` is the geometry block (which can also include direct measurements).

    :math:`f` is the vector of unknowns :math:`f = [1/c_0, u_0/c_0^2, v0/c_0^2]`

    :math:`b` are observed travel times from travelTimeExtractor

    The background flow if found through the solution:

    .. math::

        f = (G^{-1} * G)^T (G^{-1} * b)

    :math:`G` = [I, 3] (I==nMics*nSpeakers)
    :math:`f` = [3] (see above)
    :math:`f` = I (travel times for a frame)

    Attributes:
        atarray (object): The acoustic travel time array.
        directMeasurements (xr.DataArray): The direct measurements for the linear system.
        regularize (bool): A flag indicating whether to apply regularization. Default is False.
        measuredTravelTime (xr.DataArray): The measured travel time for a frame.
        includeCollocated (bool): A flag indicating whether to include collocated elements. Default is False.
    """

    atarray: object = None
    directMeasurements: xr.DataArray = None
    regularize: bool = False
    measuredTravelTime: xr.DataArray = None
    constants: object = None
    # includeCollocated: bool = False

    def __post_init__(self):
        """
        The geometry block contains all of the path orientation data for the AT array. It should have a size that corresponds to the number of acoustic travel paths, L, the dimensionality of the interrogation area (i.e., 2D or 3D) and a unity vector.
        """

        po = xr.DataArray(
            self.atarray.pathOrientation.values,
            coords={"pathID": self.measuredTravelTime.pathID},
        )
        pl = xr.DataArray(
            self.atarray.pathLength.values,
            coords={"pathID": self.measuredTravelTime.pathID},
        )

        self.ds = xr.Dataset(
            data_vars={
                "measuredTravelTime": self.measuredTravelTime,
                "pathOrientation": po,
                "pathLength": pl,
            }
        )

    def executeProcess(self):
        """
        Executes all methods of the LinearSystem in order. This includes building the process block, collecting the observation block, solving the system, and extracting the bulk values.
        """
        self.buildProcessBlock()
        self.collectObservationBlock()
        self.solve()
        self.extractBulkValues()

    def buildProcessBlock(self):
        """
        Builds the full matrix block that contains the geometry and auxiliary linear system considerations like direct measurements and model parameters for regularized systems. If direct measurements are included, G becomes a block Toeplitz structure. If regularization is included, G is augmented with model parameters.
        """

        arrayGeom = (
            xr.Dataset(
                data_vars={
                    "ones": self.atarray.pathOrientation / self.atarray.pathOrientation,
                    "ncos": -np.cos(self.atarray.pathOrientation),
                    "nsin": -np.sin(self.atarray.pathOrientation),
                },
            ).to_array()
        ).transpose()

        self.ds["arrayGeom"] = arrayGeom
        self.ds.arrayGeom.attrs = {
            "description": "Array geometry block for linear system",
            "units": "[- radians radians]",
        }

        # TODO concatenate with direct observations and parameter estimation from regularization
        # D = self.directMeasurements[None, None]
        # self.G = np.concatenate((self.arrayGeom, D), axis=0)
        # self.G = self.arrayGeom

    def collectObservationBlock(self):
        """
        Collects the observations block (b) which includes directly observed travel times. The block may also include direct measurements and model parameters measurements.
        """
        # Dimensional info
        # _, _, nFrames = self.detectedSignalTimes.shape

        # detected signal travel times

        # TODO concatenate with direct observations and parameter estimation from regularization
        observationalData = (
            self.ds.measuredTravelTime / self.ds.pathLength
        ).transpose()

        self.ds["observationalData"] = observationalData

        self.ds.observationalData.attrs = {
            "description": "travel time over path length",
            "units": "s/m",
        }

    def solve(self):
        """
        Solves the linear system assembled from the matrix and observation blocks.
        """
        # TODO it would be much better if this linear algebra could be done within the xarray framework to prevent mixup of dimensions.
        f = (
            np.linalg.pinv(self.ds.arrayGeom.to_numpy())
            @ self.ds.observationalData.to_numpy()
        )

        self.ds["variableBlock"] = xr.DataArray(
            data=f,
            coords={
                "component": np.arange(3),
                "frame": self.ds.observationalData.frame,
            },
            attrs={"description": "Unknown variables from linear system"},
        )

    def extractBulkValues(self):
        """
        Extracts the bulk velocity components and speed of sound from the result of the linear system.
        """
        bulkVals = xr.Dataset(
            data_vars={
                "c": 1 / self.ds.variableBlock[0, :],
                "u": self.ds.variableBlock[1, :] / self.ds.variableBlock[0, :] ** 2,
                "v": self.ds.variableBlock[2, :] / self.ds.variableBlock[0, :] ** 2,
            },
        )

        bulkVals.c.attrs = {
            "description": "Laplace speed of sound from AT array",
            "units": "m/s",
        }
        bulkVals.u.attrs = {
            "description": "East-West component of velocity from AT array",
            "units": "m/s",
        }
        bulkVals.v.attrs = {
            "description": "North-South component of velocity from AT array",
            "units": "m/s",
        }

        bulkVals["T"] = bulkVals["c"] ** 2 / (self.constants.gamma * self.constants.Ra)

        for key, value in bulkVals.items():
            self.ds[key] = value

    ## utility functions
    def to_netcdf(self, filePath) -> None:
        utils.to_netcdf(self.ds.unstack(), filePath)

    @classmethod
    def from_netcdf(cls, filePath):
        cls.ds = utils.from_netcdf(filePath)
        cls.ds = utils.stackDimension(
            cls.ds,
            dim="pathID",
            stackingDims=["spk", "mic"],
            dropna=True,
        )
        return cls

    def to_pickle(self, file_path):
        utils.save_to_pickle(self, file_path)

    @classmethod
    def from_pickle(cls, file_path):
        obj = utils.load_from_pickle(file_path)
        return obj

    def describe(self):
        return utils.describe_dataset(self.ds)
