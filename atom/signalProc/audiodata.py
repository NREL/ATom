from dataclasses import dataclass, field
import pandas as pd
import numpy as np
import xarray as xr
from atom import utils


@dataclass
class AudioData:
    """
    Class to handle audio data. Includes attributes related to the audio signals
    and methods for loading data, checking data validity, extracting reference signals, and exporting/importing data.

    Attributes:
        samplingFrequency (float): The frequency at which the audio data was sampled, default is 20000 Hz.
        deltaT (float): Time increment value, default is 0.05 seconds.
        recordTimeDuration (float): The duration of the audio recording, default is 0.5 seconds.
        recordTimeDelta (float): The time difference between recordings, default is 0.00005 seconds.
        recordLength (int): The length of the audio record, default is 10000.
        nFrames (int): The number of frames in the audio data, default is 120.
        chirpTimeDuration (float): The duration of the chirp signal, default is 0.0058 seconds.
        chirpRecordLength (int): The length of the chirp record, default is 116 samples.
        chirpCentralFrequency (float): The central frequency of the chirp, default is 1200 Hz.
        chirpBandwidth (float): The bandwidth of the chirp, default is 700 Hz.
        speakerSignalEmissionTime (np.ndarray): Array of speaker signal emission times.
        nMics (int): The number of microphones, default is 8.
        nSpeakers (int): The number of speakers, default is 8.
        windowHalfWidth (float): The half-width of the window, default is 0.01 seconds.
    """

    samplingFrequency: float = 20000
    deltaT: float = 0.05
    recordTimeDuration: float = 0.5
    recordTimeDelta: float = 0.00005
    recordLength: int = 10000
    nFrames: int = 120
    chirpTimeDuration: float = 0.0058
    chirpRecordLength: int = 116  # samples
    chirpCentralFrequency: float = 1200
    chirpBandwidth: float = 700
    speakerSignalEmissionTime: np.ndarray = field(
        default_factory=lambda: np.array(
            [0.1240, 0.1040, 0.2040, 0.0, 0.1600, 0.2000, 0.0400, 0.1440]
        )
    )
    nMics: int = 8
    nSpeakers: int = 8
    windowHalfWidth: float = 0.01

    def __post_init__(self):
        # generate list of attributes to pass to xarray
        coords = {
            "time": pd.TimedeltaIndex(
                np.arange(self.recordLength) / self.samplingFrequency, unit="S"
            ),
            "mic": np.arange(self.nMics),
            "frame": np.arange(self.nFrames),
        }
        attrs = {
            "description": "Signals recorded by microphones",
            "samplingFrequency": self.samplingFrequency,
            "deltaT": self.deltaT,
            "recordTimeDuration": self.recordTimeDuration,
            "recordTimeDelta": self.recordTimeDelta,
            "recordLength": self.recordLength,
            "nMics": self.nMics,
            "nFrames": self.nFrames,
            "chirpCentralFrequency": self.chirpCentralFrequency,
            "chirpBandwidth": self.chirpBandwidth,
            "nSpeakers": self.nSpeakers,
            "nMics": self.nMics,
        }

        speakerSignalEmissionTime = xr.DataArray(
            data=self.speakerSignalEmissionTime,
            coords={"spk": np.arange(self.nSpeakers)},
            name="speakerSignalEmissionTime",
            attrs={"description": "Speaker signal emission time", "units": "seconds"},
        )
        data_vars = None

        ds = utils.build_xarray(data_vars=data_vars, coords=coords, attrs=attrs)
        self.ds = ds.astype(float)

        self.ds["speakerSignalEmissionTime"] = speakerSignalEmissionTime

    def loadData(self, dataPath, keepSpkData=False):
        """
        Load the main data from the specified path.

        Args:
            dataPath (str): Path to microphone and speaker data.
            keepSpkData (bool, optional): Flag to indicate whether the full record of speaker data should be kept,
                or just a reference signal. Default is False.
        """
        colnames = ["s{}".format(x) for x in range(8)] + [
            "m{}".format(x) for x in range(8)
        ]

        mainDat = pd.read_csv(
            dataPath,
            header=3,
            delimiter="\t",
            names=colnames,
        )

        ## Mic data
        columns = [x for x in mainDat.columns if "m" in x]
        self._checkData(mainDat[columns], self.nMics)

        micData = np.transpose(
            np.reshape(
                mainDat[columns].values,
                (
                    self.recordLength,
                    self.nFrames,
                    self.nMics,
                ),
                order="F",
            ),
            axes=[0, 2, 1],
        )

        self.ds["micData"] = xr.DataArray(
            data=micData,
            coords={
                "time": self.ds.time,
                "mic": self.ds.mic,
                "frame": self.ds.frame,
            },
            name="micData",
        )

        ## Speaker data
        columns = [x for x in mainDat.columns if "s" in x]
        self._checkData(mainDat[columns], self.nSpeakers)

        spkData = np.transpose(
            np.reshape(
                mainDat[columns].values,
                (self.recordLength, self.nSpeakers, self.nFrames),
                order="F",
            ),
            axes=[0, 2, 1],
        )

        # extract reference signal from speaker data
        self.getReferenceSignal(spkData)

        # keep full record of speaker data?
        if keepSpkData:
            self.ds["spkData"] = xr.DataArray(
                data=spkData[: self.recordLength, self.nSpeakers, self.nFrames],
                coords=self.ds.coords,
                name="spkData",
            )

    def _checkData(
        self,
        data,
        nChannels,
    ):
        """
        Private method to sanity check data size/shape against config inputs.

        Args:
            data (numpy.ndarray): Data to be checked.
            nChannels (int): Number of channels as per config.
        """
        assert self.recordLength * self.nFrames * nChannels == np.prod(
            data.shape
        ), "Specified record length does not match data."

    def getReferenceSignal(self, spkData):
        """
        Extract a reference signal from one speaker.

        Args:
            spkData (numpy.ndarray): Speaker data from which the reference signal needs to be extracted.
        """

        windowHalfWidth_index = int(
            np.round(self.windowHalfWidth * self.samplingFrequency)
        )
        eta_index = int(
            (self.speakerSignalEmissionTime[0] + self.chirpTimeDuration / 2)
            * self.samplingFrequency
        )
        refSig = spkData[
            eta_index - windowHalfWidth_index : eta_index + windowHalfWidth_index,
            0,
            0,
        ]

        attrs = {
            "description": "Reference chirp signal",
            "chirpTimeDuration": self.chirpTimeDuration,
            "chirpRecordLength": self.chirpRecordLength,
            "chirpCentralFrequency": self.chirpCentralFrequency,
            "chirpBandwidth": self.chirpBandwidth,
            "windowHalfWidth": self.windowHalfWidth,
        }

        self.ds["refSig"] = xr.DataArray(
            data=refSig,
            coords={
                "time": pd.TimedeltaIndex(
                    np.arange(2 * windowHalfWidth_index) / self.samplingFrequency,
                    unit="S",
                )
            },
            name="refSig",
            attrs=attrs,
        ).dropna(dim="time")

    ## utility functions
    def to_netcdf(self, filePath) -> None:
        utils.to_netcdf(self.ds, filePath)

    @classmethod
    def from_netcdf(self, filePath) -> None:
        self.ds = utils.from_netcdf(self.ds, filePath)

    def to_pickle(self, file_path):
        utils.save_to_pickle(self, file_path)

    @classmethod
    def from_netcdf(cls, filePath):
        cls.ds = utils.from_netcdf(filePath)
        return cls

    def describe(self) -> pd.DataFrame:
        return utils.describe_dataset(self.ds)
