import numpy as np
import xarray as xr
from atom import utils


class PathSetup:
    """
    This class is responsible for setting up path integral calculations for a given set of speakers and microphones.

    Attributes:
        atarray (object): Contains the speakers and microphones data.
        ds (xr.Dataset): An xarray Dataset to store the calculations. Initially an empty dataset.
    """

    def __init__(self, atarray):
        """
        Constructs all the necessary attributes for the PathSetup object.

        Parameters:
            atarray (object): The data of the speakers and microphones.
        """
        self.atarray = atarray

        # Initialize the xarray dataset
        self.ds = xr.Dataset()

    def setupPathIntegrals(self):
        """
        Sets up path integral calculations for a given set of speakers and microphones.

        Notes:
            This method does not return any value, but modifies the class instance by adding a new attribute 'self.ds' which is
            an xarray Dataset containing the information about the paths, points for integrations, and Euclidean distances between successive points.

            It expects that the following attributes are pre-defined in the instance:
            'atarray.ds.speakerLocations', 'atarray.ds.micLocations'.

            The total number of points for integration along each path is hard-coded as 250.
        """
        # path start and stop coordinates
        self.ds["spks"] = self.atarray.ds.speakerLocations
        self.ds["mics"] = self.atarray.ds.micLocations

        paths = (
            self.ds.to_array()
            .stack(pathID=("spk", "mic"))
            .transpose("pathID", "variable", "coord")
        )
        paths = paths - paths.mean(dim="pathID")

        # exclude paths for co-located speakers and mics
        self.ds["paths"] = xr.DataArray(
            data=paths.where(paths.mic != paths.spk).dropna(dim="pathID"),
            # coords={self.ds.pathID, self.ds.coord},
        )

        # Define points for path integrations (using Simpson integrals)
        ## number of points along each path
        self.ds.attrs = dict(npaths=self.ds["paths"].shape[0], npts=250)
        ## coordinates of points for interpolation
        self.ds["points"] = xr.DataArray(
            data=np.array(
                [
                    np.array(
                        [
                            np.linspace(
                                paths[ii, 0, 0], paths[ii, 1, 0], num=self.ds.npts
                            ),
                            np.linspace(
                                paths[ii, 0, 1], paths[ii, 1, 1], num=self.ds.npts
                            ),
                        ]
                    ).T
                    for ii in range(self.ds.npaths)
                ]
            ),
            coords={
                "pathID": self.ds.pathID,
                "pointID": np.arange(self.ds.npts),
                "coord": self.ds.coord[:-1],
            },
        )

        ## delta along interpolated paths
        pointDifferences = np.diff(self.ds.points[:, :2, :], axis=1).squeeze()
        ## Calculate the Euclidean distances between successive points
        self.ds["dx"] = xr.DataArray(
            data=np.sqrt((pointDifferences[:, :-1] ** 2).sum(axis=-1)),
            coords={"pathID": self.ds.pathID},
        )

        # Format path orientation information
        po = self.atarray.ds.pathOrientation.stack(pathID=("spk", "mic"))
        # add to dataset
        self.ds["pathOrientation"] = po.where(po.mic != po.spk).dropna(dim="pathID")

    def getDataset(self):
        """
        Returns the xarray Dataset representing the path integrals.
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
