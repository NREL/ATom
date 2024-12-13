import numpy as np
import xarray as xr
from atom import utils


class ModelGrid:
    """
    A class that represents a model grid for TDSI solution.

    Attributes:
        nModelPointsX (int): Number of points in the x-direction.
        nModelPointsY (int): Number of points in the y-direction.
        modelLimsX (Tuple[float, float]): Lower and upper bounds for the x-direction.
        modelLimsY (Tuple[float, float]): Lower and upper bounds for the y-direction.

    Methods:
    buildModelGrid(): Generates a model grid using the given parameters.
    getDataset(): Returns the xarray Dataset representing the model grid.
    """

    def __init__(self, nModelPointsX, nModelPointsY, modelLimsX, modelLimsY):
        self.nModelPointsX = nModelPointsX
        self.nModelPointsY = nModelPointsY
        self.modelLimsX = modelLimsX
        self.modelLimsY = modelLimsY

        self.ds = xr.Dataset()

    def buildModelGrid(self) -> None:
        """
        Generates a model grid using the given parameters.

        This method creates a model grid as a 2D array, where 'x' and 'y' values are equally spaced based on the input parameters.
        The grid is then saved as an xarray DataArray in the 'modelGrid' attribute of the instance.

        Note: This method does not return any value, but directly modifies the 'modelGrid' attribute of the class instance.
        """

        x = np.linspace(*self.modelLimsX, self.nModelPointsX)
        y = np.linspace(*self.modelLimsY, self.nModelPointsY)

        modelGrid = np.meshgrid(x, y)
        self.ds["modelGrid"] = xr.DataArray(
            data=np.array(modelGrid),
            coords={
                "variable": ["x", "y"],
                "x": x,
                "y": y,
            },
        )
        self.ds.attrs = {
            "description": "Model grid for TDSI solution",
            "unit": "m",
            "nx": len(self.ds.x),
            "ny": len(self.ds.y),
        }

        tmp = self.ds[["x", "y"]].stack(modelXY=["x", "y"])
        self.ds["modelXY"] = tmp.modelXY

    def getDataset(self):
        """
        Returns the xarray Dataset representing the model grid.
        """
        return self.ds

    ## utility functions
    def to_netcdf(self, filePath) -> None:
        utils.to_netcdf(self.ds, filePath)

    @classmethod
    def from_netcdf(self, filePath) -> None:
        self.ds = utils.from_netcdf(self.ds, filePath)

    def to_pickle(self, file_path):
        utils.save_to_pickle(self, file_path)

    @classmethod
    def from_pickle(cls, file_path):
        obj = utils.load_from_pickle(file_path)
        return obj

    def describe(self):
        return utils.describe_dataset(self.ds)
