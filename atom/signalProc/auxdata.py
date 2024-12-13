from dataclasses import dataclass, field
import pandas as pd
import numpy as np
import xarray as xr
from atom import utils


@dataclass
class AuxData:
    """
    This class represents auxiliary data, including sonic anemometer and TH probe data.

    Attributes:
        samplingFrequency (float): The sampling frequency of the data.
        recordTimeDuration (float): The time duration of the data in seconds.
        recordTimeDelta (float): The time delta between each sample in seconds.
        recordLength (int): The length of the data.
        variables (list): The variables to include in the data.
        sonicAnemometerOrientation (float): The orientation of the sonic anemometer in degrees.
        windowType (str, optional): The type of window to apply to the velocity signal when calculating the velocity spectrum. Defaults to "hanning".
        threshold (float, optional): The threshold to use when calculating the integral length scale. Defaults to 0.5.
    """

    samplingFrequency: float
    recordTimeDuration: float
    recordTimeDelta: float
    recordLength: int
    variables: field(default_factory=list)  # type: ignore
    sonicAnemometerOrientation: float
    windowType: str = "hann"
    threshold: float = 0.5

    def __post_init__(self):
        """
        Initialize an `AuxData` object by building a `xarray.Dataset` with default values for the data variables.
        """
        # generate list of attributes to pass to xarray
        attrs = self.__dict__.copy()
        _ = attrs.pop("variables")

        coords = {
            "time": pd.TimedeltaIndex(
                np.arange(self.recordLength) * self.recordTimeDelta, unit="S"
            )
        }

        data_vars = {var: None for var in self.variables}

        ds = utils.build_xarray(data_vars=data_vars, coords=coords, attrs=attrs)
        self.ds = ds.astype(float)

    def loadData(self, auxdatapath, applySonicOrientation=True):
        """
        Load auxiliary data from specified file path.

        Args:
            auxdatapath (str): path to the auxiliary data file.
            applySonicOrientation (bool, optional): flag to indicate whether to reorient the u and v signals to match the orientation of the sonic anemometer in the field. Default is True.

        Returns:
            None
        """
        colnames = self.variables
        auxdata = pd.read_csv(auxdatapath, header=3, delimiter="\t", names=colnames)

        # Calculate wind speed and direction
        auxdata["WS"] = np.sqrt(
            auxdata["ux"] ** 2 + auxdata["uy"] ** 2 + auxdata["uz"] ** 2
        )
        auxdata["WD"] = -np.degrees(np.arctan2(auxdata["uy"], auxdata["ux"])) + 270

        # Format time axis
        auxdata["time"] = pd.timedelta_range(
            start=0, periods=len(auxdata), freq="0.05S"
        )
        auxdata.set_index("time", inplace=True)
        # auxdata = auxdata.iloc[: self.recordLength]

        if applySonicOrientation:
            rotatedVel = self._rotateHorizontalVelocity(
                auxdata[["ux", "uy"]], angtype="deg"
            )
        auxdata = pd.concat([auxdata, rotatedVel], axis=1)

        # Merge with the existing xarray dataset
        self.ds = xr.merge([self.ds, auxdata.to_xarray()])
        self._addChannelAttrs()

    def autoCorrelation(
        self, variables=["u", "v", "uz", "T", "H"], lags=None
    ) -> xr.DataArray:
        """
        Calculate the velocity autocorrelation from a turbulent velocity signal.

        Returns:
            xarray.DataArray: The velocity autocorrelation, with dimensions 'lag' and 'vector_component'.
        """

        def acorr(x, lags):
            x = x - x.mean()
            corr = np.correlate(x, x, "full")[len(x) - 1 :] / (np.var(x) * len(x))

            return corr[:lags]

        if lags == None:
            lags = len(self.ds.time)

        for variable in variables:

            x = self.ds[variable].values

            rho = xr.DataArray(
                data=acorr(x, lags=lags),
                coords={"time": self.ds.time},
                attrs={
                    "description": f"Autocorrelation coefficient of {variable}",
                    "units": "None",
                    "Note": "time dimension represents the time lag between observations",
                },
            )

            self.ds[f"rho_{variable}"] = rho

    def integral_scale(
        self,
        variables=["u", "v", "uz", "T", "H"],
    ) -> None:
        """
        Calculate the integral scale from an autocorrelation function.

        Args:
            autocorrelation (xr.DataArray): The autocorrelation function, with dimensions 'time_lag' and 'vector_component'.
            dt (xr.DataArray): The time lag, with dimension 'time_lag'.

        Returns:
            float: The integral scale.
        """

        vars = [x for x in self.ds.data_vars if "S_" in x]

        for variable in variables:

            spec = f"S_{variable}"
            integralScale = (self.ds[spec] / self.ds[spec].frequency).where(
                self.ds[spec].frequency > 0.0
            ).dropna(dim="frequency").integrate(coord="frequency") / self.ds[
                spec
            ].where(
                self.ds[spec].frequency > 0.0
            ).dropna(
                dim="frequency"
            ).integrate(
                coord="frequency"
            )
            integralScale = integralScale.values

            self.ds.attrs[f"tau_{variable}"] = integralScale.max()
            self.ds.attrs[f"L_{variable}"] = (
                self.ds[variable].mean(dim="time").values * integralScale.max()
            )

    def calculateSpectra(
        self,
        nperseg: int = 256,
        nfft: int = 1024,
        variables=["u", "v", "uz", "T", "H"],
    ) -> None:
        """
        Calculate the velocity spectrum from a turbulent velocity signal using the FFT.

        Returns:
            xarray.DataArray: The velocity spectrum, with dimensions 'frequency', 'vector_component', and 'spatial_direction'.
        """
        from scipy.signal import welch

        for variable in variables:

            frequency, spectrum = welch(
                self.ds[variable],
                fs=self.samplingFrequency,
                nperseg=nperseg,
                nfft=nfft,
            )
            if variable == "T":
                units = "K^2/s"
            else:
                units = "m^2/s"

            spectrum = xr.DataArray(
                spectrum,
                dims=("frequency"),
                coords={
                    "frequency": frequency,
                },
                attrs={
                    "description": f"Spectrum of {variable} using Welch's method",
                    "units": units,
                    "sampling frequency [Hz]": self.samplingFrequency,
                    "window": self.windowType,
                    "samples per segment": nperseg,
                    "zero-padded record length": nfft,
                },
            )

            self.ds[f"S_{variable}"] = spectrum

    def _rotateHorizontalVelocity(self, velocityData, angtype="deg"):
        """Rotate horizontal velocity components reported by the sonic into the reference frame

        Args:
            auxdata (pd.DataFrame): Sonic anemometer and TH probe dta
            theta (float): Orientation angle of sonic anemometer
            angtype (str, optional): Units of orientation angle. Defaults to "deg".

        Returns:
            pd.DataFrame: Horizontal components of velocity in reference frame
        """
        return pd.DataFrame(
            data=np.dot(
                self._rotation_matrix(self.sonicAnemometerOrientation, angtype=angtype),
                velocityData[["ux", "uy"]].T,
            ).T,
            index=velocityData.index,
            columns=["u", "v"],
        )

    def _rotation_matrix(self, theta, angtype="rad"):
        """Generate 2D rotation matrix

        Args:
            theta (float): angle of rotation
            angtype (str, optional): units of rotation angle. Defaults to "rad".

        Returns:
            np.array: 2D rotation matrix
        """
        if angtype == "deg":
            theta = np.radians(theta)
        rotmat = np.squeeze(
            np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        )
        return rotmat

    def _addChannelAttrs(self) -> None:
        """
        add attrs to each xr.dataArray in self.ds
        """
        ## attributes
        channelAttrs = {
            "ux": {
                "description": "Longitudinal component of velocity from sonic",
                "unit": "m/s",
            },
            "uy": {
                "description": "Transverse component of velocity from sonic",
                "unit": "m/s",
            },
            "uz": {
                "description": "Vertical component of velocity from sonic",
                "unit": "m/s",
            },
            "c": {"description": "Adiabatic speed of sound", "unit": "m/s"},
            "T": {"description": "Ambient temperature", "unit": "degC"},
            "H": {"description": "Relative Humidity", "unit": "%"},
            "WS": {"description": "Wind speed", "unit": "m/s"},
            "WD": {"description": "wind direction", "unit": "deg"},
            "u": {
                "description": "East-west component of velocity from sonic",
                "unit": "m/s",
            },
            "v": {
                "description": "North-south component of velocity from sonic",
                "unit": "m/s",
            },
        }

        for varname in self.ds.data_vars:
            self.ds[varname].attrs = channelAttrs[varname]

    ## utility functions
    def to_netcdf(self, filePath) -> None:
        utils.to_netcdf(self.ds, filePath)

    @classmethod
    def from_netcdf(cls, filePath):
        cls.ds = utils.from_netcdf(filePath)
        return cls

    def to_pickle(self, file_path):
        utils.save_to_pickle(self, file_path)

    @classmethod
    def from_pickle(cls, file_path):
        obj = utils.load_from_pickle(file_path)
        return obj

    def describe(self) -> pd.DataFrame:
        return utils.describe_dataset(self.ds)

    # TODO: background flow data:
    # mean or low-pass filter u, v, c0
    # pass to background flow module
