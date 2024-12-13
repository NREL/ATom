from dataclasses import dataclass, asdict
import numpy as np
import xarray as xr
from atom import utils

logger = logging.getLogger(__name__)


@dataclass
class ATArray:  # ordered=True
    """
    A class to hold information about the microphone and speaker arrays and their interactions.

    Attributes:
        nMics (int): Number of microphones.
        micLocations (np.ndarray): Location coordinates of microphones.
        nSpeakers (int): Number of speakers.
        speakerLocations (np.ndarray): Location coordinates of speakers.
        nTravelPaths (int): Number of travel paths.
        hardwareSignalDelays (np.ndarray): Time-delay hardware latencies.
        npsSpeakerDelays (np.ndarray): Time-delay from non-point source speakers.
    """

    nMics: int
    micLocations: np.ndarray
    nSpeakers: int
    speakerLocations: np.ndarray
    nTravelPaths: int
    hardwareSignalDelays: np.ndarray
    npsSpeakerDelays: np.ndarray
    includeCollocated: bool = False
    nIntegralPoints: int = 250

    # Define xr.DataArrays and attributes for all input data
    def __post_init__(self):
        mics = np.arange(self.nMics)
        spks = np.arange(self.nSpeakers)
        coordinates = ["northing", "easting", "elevation"]

        self.ds = xr.Dataset()

        # instrument locations
        self.ds["micLocations"] = xr.DataArray(
            data=np.array(self.micLocations),
            coords={"mic": mics, "coord": coordinates},
            attrs={
                "description": "coordinates of microphone mounting locations",
                "units": "meters",
            },
            name="micLocations",
        )
        self.ds["speakerLocations"] = xr.DataArray(
            data=np.array(self.speakerLocations),
            coords={"spk": spks, "coord": coordinates},
            attrs={
                "description": "coordinates of speaker mounting locations",
                "units": "meters",
            },
            name="speakerLocations",
        )
        # Signal delays and offsets
        self.ds["hardwareSignalDelays"] = xr.DataArray(
            data=np.array(self.hardwareSignalDelays),
            coords={
                "spk": spks,
                "mic": mics,
            },
            attrs={"description": "time-delay hardware latency", "units": "seconds"},
            name="hardwareSignalDelays",
        )
        self.ds["npsSpeakerDelays"] = xr.DataArray(
            data=np.array(self.npsSpeakerDelays),
            coords={
                "spk": spks,
                "mic": mics,
            },
            attrs={
                "description": "time-delay from non-point source speakers",
                "units": "seconds",
            },
            name="npsSpeakerDelays",
        )

        self.ds.attrs = {
            "nMics": self.nMics,
            "nSpeakers": self.nSpeakers,
            "nTravelPaths": self.nTravelPaths,
            "nIntegralPoints": self.nIntegralPoints,
        }

        ## Derive path information
        self.calcPathInfo()
        ## Check the number of ray paths
        altNpaths = self.nMics * self.nSpeakers
        assert (
            altNpaths == self.nTravelPaths
        ), "Specified number of travel paths does not match nMics * nSpeakers."

        if self.includeCollocated is False:
            self.ds = self.stackPathID(
                dim="pathID", stackingDims=["spk", "mic"], dropna=True
            )
            # self.ds = self.ds.where(self.ds.spk != self.ds.mic).dropna(dim="pathID")

    def calcPathInfo(self):
        """
        Calculate path vectors, lengths, and orientations based on microphone and speaker locations.

        The method calculates:
        - Path vectors: Vector from speaker to microphone.
        - Path lengths: Distance from speaker to microphone.
        - Path orientations: Angular orientation of the path.
        """
        # offsets in x and y directions
        delx = (
            self.ds.micLocations[:, 1].values[None, :]
            - self.ds.speakerLocations[:, 1].values[:, None]
        )
        dely = (
            self.ds.micLocations[:, 0].values[None, :]
            - self.ds.speakerLocations[:, 0].values[:, None]
        )

        micLocs = self.ds.micLocations - self.ds.micLocations.mean("mic")
        spkLocs = self.ds.speakerLocations - self.ds.speakerLocations.mean("spk")
        # trim off vertical component (assume perfectly 2D for now)
        # TODO update for 3D
        micLocs = micLocs[:, :2]
        spkLocs = spkLocs[:, :2]

        # paths
        paths = (
            self.ds[["speakerLocations", "micLocations"]]
            .to_array()
            .stack(pathID=("spk", "mic"))
            .transpose(
                "pathID",
                "variable",
                "coord",
            )
        )
        paths = paths - paths.mean(dim="pathID")
        self.ds["integralPaths"] = paths

        # path vectors
        self.ds["pathVector"] = xr.DataArray(
            data=np.array(
                [[dx, dy] for dx, dy in zip(delx.flatten(), dely.flatten())]
            ).reshape([self.nMics, self.nSpeakers, 2]),
            coords={"spk": self.ds.spk, "mic": self.ds.mic, "component": ["dx", "dy"]},
            attrs={
                "description": "path vector from speaker to microphone",
                "units": "meters",
            },
            name="pathVector",
        )
        # path lengths
        self.ds["pathLength"] = xr.DataArray(
            data=np.linalg.norm(self.ds.pathVector, axis=-1),
            coords={
                "spk": self.ds.spk,
                "mic": self.ds.mic,
            },
            attrs={
                "description": "path length from speaker to microphone",
                "units": "meters",
            },
            name="pathLength",
        )
        # path orientations
        self.ds["pathOrientation"] = xr.DataArray(
            data=np.arctan2(
                self.ds.pathVector.values[:, :, 1], self.ds.pathVector.values[:, :, 0]
            ),
            coords={
                "spk": self.ds.spk,
                "mic": self.ds.mic,
            },
            attrs={
                "description": "path angular orientation",
                "units": "radians",
            },
            name="pathOrientation",
        )

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
        paths = self.ds["integralPaths"]
        # Define points for path integrations (using Simpson integrals)
        ## coordinates of points for interpolation
        self.ds["integralPoints"] = xr.DataArray(
            data=np.array(
                [
                    np.array(
                        [
                            np.linspace(
                                paths[ii, 0, 0],
                                paths[ii, 1, 0],
                                num=self.ds.nIntegralPoints,
                            ),
                            np.linspace(
                                paths[ii, 0, 1],
                                paths[ii, 1, 1],
                                num=self.ds.nIntegralPoints,
                            ),
                        ]
                    ).T
                    for ii in range(len(self.ds.pathID))
                ]
            ),
            coords={
                "pathID": self.ds.pathID,
                "pointID": np.arange(self.ds.nIntegralPoints),
                "coord": self.ds.coord[:-1],
            },
        )

        ## delta along interpolated paths
        pointDifferences = np.diff(self.ds.integralPoints[:, :2, :], axis=1).squeeze()
        ## Calculate the Euclidean distances between successive points
        self.ds["integraldx"] = xr.DataArray(
            data=np.sqrt((pointDifferences[:, :-1] ** 2).sum(axis=-1)),
            coords={"pathID": self.ds.pathID},
        )

    def excludeColocated(
        self, dim: str = "pathID", stackingDims: list = ["spk", "mic"]
    ):
        ds = self.ds.stack({dim: stackingDims})
        self.ds = ds.where(ds[stackingDims[0]] != ds[stackingDims[1]]).dropna(dim=dim)

    def plotSpkMicLocations(self, ax=None, addLabels: bool = True):
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots()

        spkLocs = self.ds.speakerLocations.unstack().median(dim="mic")
        spkLocs -= spkLocs.mean(dim="spk")
        micLocs = self.ds.micLocations.unstack().median(dim="spk")
        micLocs -= micLocs.mean(dim="mic")

        ax.scatter(
            spkLocs.sel(coord="easting"), spkLocs.sel(coord="northing"), label="Speaker"
        )
        ax.scatter(
            micLocs.sel(coord="easting"), micLocs.sel(coord="northing"), label="Mic"
        )
        for ii in range(self.ds.nSpeakers):
            ax.text(
                0.85 * spkLocs.sel(coord="easting")[ii],
                0.85 * spkLocs.sel(coord="northing")[ii],
                f"Tower {ii}",
                color=".25",
                horizontalalignment="center",
                verticalalignment="center",
                fontsize=8,
            )
        ax.legend()
        ax.set_xlabel("Easting [m]")
        ax.set_ylabel("Northing [m]")

        return ax

    def plotPaths(self, ax=None, c=None, lw=1, alpha=0.75):
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots()

        if c is None:
            c = ".25"
            for ip, path in enumerate(self.ds.integralPaths.isel(coord=[0, 1])):
                ax.plot(
                    path.values[:, 1],
                    path.values[:, 0],
                    lw=lw,
                    c=c,
                    alpha=alpha,
                )

        else:
            for ip, path in enumerate(self.ds.integralPaths.isel(coord=[0, 1])):
                ax.plot(
                    path.values[:, 1],
                    path.values[:, 0],
                    lw=lw,
                    c=c[ip, :],
                    alpha=alpha,
                )
                x = (
                    0.8 * path.sel(coord="easting")[0]
                    + 0.2 * path.sel(coord="easting")[1]
                )
                y = (
                    0.8 * path.sel(coord="northing")[0]
                    + 0.2 * path.sel(coord="northing")[1]
                )
                ax.text(
                    x,
                    y,
                    f"P{ip}",
                    color=".25",
                    horizontalalignment="center",
                    verticalalignment="center",
                    fontsize=8,
                )

        ax.set_xlabel("Easting [m]")
        ax.set_ylabel("Northing [m]")

        return ax

    def to_netcdf(self, filePath) -> None:
        utils.to_netcdf(self.ds, filePath)

    @classmethod
    def from_netcdf(cls, filePath):
        cls.ds = utils.from_netcdf(filePath)
        return cls

    def stackPathID(
        self,
        dim: str = "pathID",
        stackingDims: list = ["spk", "mic"],
        dropna: bool = True,
    ):
        return utils.stackDimension(
            self.ds, dim=dim, stackingDims=stackingDims, dropna=dropna
        )
