import xarray as xr
from scipy.interpolate import RegularGridInterpolator
import numpy as np
from atom import utils


class LESData:
    def __init__(
        self,
        loadPath=None,
        atarray: xr.Dataset = None,
        constants: xr.Dataset = None,
    ):
        self.loadPath = loadPath
        self.atarray = atarray
        self.constants = constants

        self.load_data()
        self.ds.attrs = dict(
            datapath=self.loadPath,
            gamma=self.constants.gamma,
            Ra=self.constants.Ra,
        )

    def load_data(self):
        """#TODO add validator"""
        self.ds = xr.load_dataset(self.loadPath)

    def get_bulk_values(self):
        self.ds["uBulk"] = self.ds.u.mean(dim=["x", "y"])
        self.ds["vBulk"] = self.ds.v.mean(dim=["x", "y"])
        self.ds["TBulk"] = self.ds.T.mean(dim=["x", "y"])

        self.ds["cBulk"] = np.sqrt(
            self.constants.gamma * self.constants.Ra * self.ds["TBulk"]
        )

    def setup_interpolators(self):
        uFluc = self.ds.u - self.ds["uBulk"]
        vFluc = self.ds.v - self.ds["vBulk"]
        TFluc = self.ds.T - self.ds["TBulk"]

        self.uinterp = RegularGridInterpolator(
            (self.ds.time, self.ds.x, self.ds.y),
            uFluc.transpose("time", "x", "y").values,
        )
        self.vinterp = RegularGridInterpolator(
            (self.ds.time, self.ds.x, self.ds.y),
            vFluc.transpose("time", "x", "y").values,
        )
        self.Tinterp = RegularGridInterpolator(
            (self.ds.time, self.ds.x, self.ds.y),
            TFluc.transpose("time", "x", "y").values,
        )

    # define a function for interpolation first (will turn this into a class)
    def interpolate_field(self, times=None):
        """
        The method interpolates LES data at given points.
        - times: times over which to interpolate velocity and temperature fields.
            Defaults to None, in which case, all times from the LES data are used.
        :return: Interpolated values at the given points.
        """

        self.get_bulk_values()
        self.setup_interpolators()

        if times is None:
            times = self.ds.time

        coords = {
            "time": times.values,
            "pathID": self.atarray.integralPoints.pathID,
            "pointID": self.atarray.integralPoints.pointID,
        }

        xpts = self.atarray.integralPoints.isel(coord=0)
        ypts = self.atarray.integralPoints.isel(coord=1)

        self.ds["uint"] = xr.DataArray(
            data=self.uinterp(
                (
                    times.values[:, None, None],
                    xpts.values[None, ...],
                    ypts.values[None, ...],
                )
            ),
            coords=coords,
        )
        self.ds["vint"] = xr.DataArray(
            data=self.vinterp(
                (
                    times.values[:, None, None],
                    xpts.values[None, ...],
                    ypts.values[None, ...],
                )
            ),
            coords=coords,
        )
        self.ds["Tint"] = xr.DataArray(
            data=self.Tinterp(
                (
                    times.values[:, None, None],
                    xpts.values[None, ...],
                    ypts.values[None, ...],
                )
            ),
            coords=coords,
        )

    def calculate_travel_time(self):
        """
        Calculate the travel time using Equation 5.

        Returns:
            float: Calculated travel time for the path including measurement error.
        """
        if "uint" not in self.ds:  # check for interpolated velocity fields
            self.interpolate_field()

        cosPathOrientation = xr.DataArray(
            data=np.cos(self.atarray.pathOrientation.values),
            coords={"pathID": self.ds.pathID},
        )
        sinPathOrientation = xr.DataArray(
            data=np.sin(self.atarray.pathOrientation.values),
            coords={"pathID": self.ds.pathID},
        )
        integraldx = xr.DataArray(
            data=self.atarray.integraldx.values, coords={"pathID": self.ds.pathID}
        )
        pathLength = xr.DataArray(
            data=self.atarray.pathLength.values, coords={"pathID": self.ds.pathID}
        )

        integrand = (
            (self.ds["cBulk"] / (2 * self.ds["TBulk"])) * self.ds["Tint"]
            + cosPathOrientation * self.ds["uint"]
            + sinPathOrientation * self.ds["vint"]
        )
        FluctuatingTravelTimes = (
            integrand.integrate(
                coord="pointID",
            )
            * integraldx
            / self.ds["cBulk"] ** 2
        )
        self.ds["fluctuatingTravelTimes"] = FluctuatingTravelTimes

        travelTimes = FluctuatingTravelTimes + (
            pathLength
            / self.ds["cBulk"]
            * (
                1
                - (
                    cosPathOrientation * self.ds["uBulk"]
                    + sinPathOrientation * self.ds["vBulk"]
                )
                / self.ds["cBulk"]
            )
        )

        self.ds["travelTimes"] = travelTimes

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
