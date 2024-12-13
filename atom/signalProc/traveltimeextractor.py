from dataclasses import dataclass
import numpy as np
from scipy import fftpack as fft
from scipy.interpolate import interp1d
from scipy.signal import sosfiltfilt, butter, find_peaks
import xarray as xr
from atom import utils


class TravelTimeExtractor:
    """
    A class to perform extraction of travel times from recorded acoustic signals.

    Attributes:
        upsampleFactor (int): The factor by which to upsample the data. Default is 10.
        backgroundFlow (str): The background flow model. Can be "homogeneous", "loglaw", "powerlaw", or "FLORIS". Default is "homogeneous".
        backgroundTemp (str): The background temperature model. Default is "homogeneous".
        filterType (str): The type of filter to use. Default is "butter".
        filterOrder (int): The order of the filter. Default is 4.
        filterScale (float): The scale of the filter. Default is 0.5.
        peakHeight (float): The height of the peak. Default is 3.25.
        peakDistance (int): The distance of the peak. Default is 200.
        filterCOM (float): The center of mass for the filter. Default is 99999.0.
        filterWidth (float): The width of the filter. Default is 4.0.
        delayOffset (float): The delay offset. Default is -0.001.
        atarray (object): The acoustic travel time array. Default is None.
        audiodata (object): The audio data. Default is None.
        auxdata (object): The auxiliary data. Default is None.
        ds (xarray.Dataset): The dataset containing the attributes of the class.
    """

    def __init__(
        self,
        configData: dict = {
            "Note": "refer to configuration yaml file for correct inputs"
        },
        atarray: xr.Dataset = None,
        audiodata: xr.Dataset = None,
        auxdata: xr.Dataset = None,
        correctSignalDelayEstimate: bool = True,
    ):

        self.atarray = atarray
        self.audiodata = audiodata
        self.auxdata = auxdata
        self.correctSignalDelayEstimate = correctSignalDelayEstimate

        # coords = {}
        ds = utils.build_xarray(data_vars=None, coords=None, attrs=configData)
        self.ds = ds.astype(float)

        # Apply travel time estimate correction
        if self.correctSignalDelayEstimate:
            # correction model values
            self.ds.attrs["delayOffset"] = list(self.ds.attrs["delayOffset"])
            amplitude, freq, phase, offset = self.ds.delayOffset

            # model dependencies
            udir = (
                np.arctan2(self.auxdata.v, self.auxdata.u).resample(time="0.5S").mean()
            ).rename({"time": "frame"})
            c0 = self.auxdata.c.resample(time="0.5S").mean().rename({"time": "frame"})
            dtheta = self.atarray.pathOrientation - udir

            # correction
            signalDelayOffset = (
                amplitude
                * self.atarray.pathLength
                * np.sin(freq * dtheta + phase)
                / c0**2
                + offset * self.atarray.pathLength
            )
            signalDelayOffset = signalDelayOffset.assign_coords(
                frame=np.arange(self.audiodata.nFrames)
            )

            self.ds["signalDelayOffset"] = signalDelayOffset

    def extractTravelTimes(
        self,
    ):
        """
        Executes the whole travel time extraction process.

        This method performs the entire process of travel time extraction in a sequential manner by calling the necessary internal methods in the right order.
        """
        self.signalETAs()
        self.filterMicData()
        self.refSig = self.audiodata.refSig
        self.findPeakCorrelations()
        self.calculateMeasuredTravelTimes()
        self.filterTravelTimes(filterIterations=self.ds.filterIterations)

    def signalETAs(self):
        """
        Calculates the expected signal arrival times based on path and group velocity.

        This method calculates the expected time of arrival of signals at each microphone using the paths and the group velocity. It takes into account the projection of the velocity vector from the sonic onto the travel paths.
        """
        expectedTravelTimes = (
            (
                self.atarray.pathLength
                / self.auxdata.c
                * (
                    1
                    - self.auxdata.u
                    * np.cos(self.atarray.pathOrientation)
                    / self.auxdata.c
                    - self.auxdata.v
                    * np.sin(self.atarray.pathOrientation)
                    / self.auxdata.c
                )
            )
            .resample(time="0.5S")
            .mean()
        )

        self.ds["expectedTravelTimes"] = xr.DataArray(
            data=expectedTravelTimes.values,
            coords={
                "pathID": self.atarray.pathID,
                "frame": self.audiodata.frame,
            },
        )

        self.ds["signalDelays"] = (
            (
                self.audiodata.speakerSignalEmissionTime
                + self.atarray.hardwareSignalDelays.unstack()
                + self.atarray.npsSpeakerDelays.unstack()
            )
            .stack(pathID=["spk", "mic"])
            .dropna(dim="pathID", how="all")
        )

        if self.correctSignalDelayEstimate:
            self.ds["signalDelays"] = (
                self.ds["signalDelays"] + self.ds["signalDelayOffset"]
            )

        # expectedSignalArrivalTimes = signal emission time + path length / expectedPathVelocity + hardware delay  + nonpoint source speaker correction [seconds]
        self.ds["expectedSignalArrivalTimes"] = (
            self.ds.expectedTravelTimes + self.ds["signalDelays"]
        )

        self.ds["expectedSignalArrivalTimes"].attrs = {
            "description": "expected arrival time of a speaker from spk to mic",
            "unit": "second",
        }

    def filterMicData(self):
        """
        Applies a filter to the microphone data.

        This method applies a selected filter to the recorded microphone data. The type of filter applied depends on the `filterType` attribute. The available options are 'fft' or 'butter'.
        """
        if self.ds.filterType == "fft":
            filterFreqLimits = np.array(
                [
                    self.audiodata.chirpCentralFrequency
                    - self.ds.filterScale * self.audiodata.chirpBandwidth,
                    self.audiodata.chirpCentralFrequency
                    + self.ds.filterScale * self.audiodata.chirpBandwidth,
                ]
            )
            micSpec = fft.fft(self.audiodata.micData, axis=0)
            freqs = fft.fftfreq(
                self.audiodata.recordLength, 1 / self.audiodata.samplingFrequency
            )
            # TODO add other filter mechanisms
            # spectra outside chirp frequency range should be zero.
            micSpec[(np.abs(freqs) < filterFreqLimits[0])] = 0
            micSpec[(np.abs(freqs) > filterFreqLimits[1])] = 0
            self.micDataFiltered = np.real(fft.ifft(micSpec, axis=0))

        elif self.ds.filterType == "butter":
            ## define sos filter
            filterFreqLimits = np.array(
                [
                    self.audiodata.chirpCentralFrequency
                    - self.ds.filterScale * self.audiodata.chirpBandwidth,
                    self.audiodata.chirpCentralFrequency
                    + self.ds.filterScale * self.audiodata.chirpBandwidth,
                ]
            )
            sos = butter(
                self.ds.filterOrder,
                filterFreqLimits,
                output="sos",
                btype="bandpass",
                fs=self.audiodata.samplingFrequency,
            )
            micDataFiltered = sosfiltfilt(sos, self.audiodata.micData, axis=0)

            self.micDataFiltered = xr.DataArray(
                data=micDataFiltered,
                coords=self.audiodata.micData.coords,
                attrs=self.audiodata.micData.attrs,
            )

    def extractSubSeries(self, keepSubData=False):
        """
        Extracts subseries of data from the audio signal.

        Parameters:
            keepSubData (bool): Flag indicating whether to keep the subseries data.
                                If set to False, the subseries data will be discarded.

        Returns:
            None
        """
        # TODO this needs to be vectorized for all mics, all received signals, and all frames.

        timeVec = np.arange(
            0, self.audiodata.recordTimeDuration, 1 / self.audiodata.samplingFrequency
        )

        # start time indices of subsamples
        # windowStarts = np.abs(self.subSampTime).argmin(axis=0)
        windowStarts = (
            (self.expectedSignalArrivalTimes - self.searchWindowWidth)
            * self.audiodata.samplingFrequency
        ).astype(int)

        windowLength = int(
            2 * self.searchWindowWidth * self.audiodata.samplingFrequency
        )

        nm, ns, nf = (
            self.audiodata.nMics,
            self.audiodata.nSpeakers,
            self.audiodata.nFrames,
        )
        # extract subsamples

        md = self.micDataFiltered.reshape(10000, -1).copy()
        ws = windowStarts.reshape(nm, ns * nf)

        self.micSubData = np.zeros([windowLength, nm, ns * nf])
        for mm in range(nm):
            for ii in range(ns * nf):
                self.micSubData[:, mm, ii] = md[:, mm].take(
                    np.arange(ws[mm, ii], ws[mm, ii] + windowLength)
                )

        self.micSubData /= self.micSubData.max(axis=0)[None, ...]
        self.micSubData = self.micSubData.reshape([windowLength, nm, ns, nf])

        # extract time indices of subsamples
        self.subSampTime = np.zeros([windowLength, nm, ns * nf])
        for mm in range(nm):
            for ii in range(ns * nf):
                self.subSampTime[:, mm, ii] = timeVec.take(
                    np.arange(ws[mm, ii], ws[mm, ii] + windowLength)
                )
        self.subSampTime = self.subSampTime.reshape([windowLength, nm, ns, nf])

    def upsampleMicData(self):
        """
        Upsample microphone subsamples, associated time vectors, and reference signals.
        """

        # time delta vector
        tDeltaVec = np.arange(
            0, 2 * self.searchWindowWidth, 1 / self.audiodata.samplingFrequency
        )
        # upsampled time delta vector
        tDeltaVecUpsample = np.arange(
            tDeltaVec[0],
            tDeltaVec[-1],
            1 / (self.audiodata.samplingFrequency * self.upsampleFactor),
        )
        # interpolate along time axis
        micInterpFunc = interp1d(tDeltaVec, self.micSubData, axis=0)
        timeInterpFunc = interp1d(tDeltaVec, self.subSampTime, axis=0)
        refSigInterpFunc = interp1d(tDeltaVec, self.audiodata.refSig)

        # overwrite signals and time with upsampled versions
        self.refSig = refSigInterpFunc(tDeltaVecUpsample).dropna(dim="time")
        self.micSubData = micInterpFunc(tDeltaVecUpsample)
        self.subSampTime = timeInterpFunc(tDeltaVecUpsample)

    def correlateRefSig(self):
        """
        Correlate microphone data around signal ETAs with reference signal.
        """
        nt, nm, ns, nf = self.micSubData.shape
        # Cross-correlation of subsamples and reference signal
        corr = np.array(
            [
                np.correlate(
                    self.micSubData[:, mm, ss, ff],
                    self.refSig,
                    mode="same",
                )
                for mm in range(nm)
                for ss in range(ns)
                for ff in range(nf)
            ]
        ).T.reshape(nt, nm, ns, nf)

        # timestamp of maximum correlation
        maxCorrLocs = np.abs(corr).argmax(axis=0)
        self.detectedSignalTimes = (
            np.array(
                [
                    self.subSampTime[maxCorrLocs[mm, ss, ff], mm, ss, ff]
                    for mm in range(nm)
                    for ss in range(ns)
                    for ff in range(nf)
                ]
            ).reshape(nm, ns, nf)
            - self.audiodata.speakerSignalEmissionTime[:, None, None]
        )

    def findPeakCorrelations(self):
        """
        Detects the peak correlations between the microphone signals and the reference signal.

        This method detects the peaks of correlation between the microphone signals and the reference signal by calculating the cross-correlation between the two signals and identifying the peaks in the correlation result.
        """
        sigETA = self.ds.expectedSignalArrivalTimes.unstack()
        detected_signal_times = xr.DataArray(
            data=np.zeros(sigETA.shape), coords=sigETA.coords
        )

        for mic in range(8):
            for frame in range(120):
                peaks, _ = find_peaks(
                    -self.micDataFiltered.isel(mic=mic, frame=frame).values,
                    height=self.ds.peakHeight,
                    distance=self.ds.peakDistance,
                )

                dmat = (
                    peaks[:, None] / self.audiodata.samplingFrequency
                    - sigETA.isel(mic=mic, frame=frame).data[None, :]
                )

                inds = np.argmin(np.abs(dmat), axis=0)
                detected_signal_times.loc[frame, :, mic] = (
                    peaks[inds] / self.audiodata.samplingFrequency
                )

        detected_signal_times = detected_signal_times.where(~sigETA.isnull())

        self.ds["detectedSignalTimes"] = detected_signal_times.stack(
            pathID=["spk", "mic"]
        ).dropna(dim="pathID", how="all")

    def calculateMeasuredTravelTimes(
        self,
    ):
        """
        Calculates the measured travel times for each speaker and microphone pair.

        This method calculates the measured travel times for each speaker and microphone pair. It considers the time delay due to the emission time, hardware signal delays and NPS speaker delays. It then compares these with the expected arrival times to determine the travel time deltas.
        """

        # Travel time is detected signal time minus emission time
        self.ds["measuredTravelTimes"] = (
            self.ds.detectedSignalTimes - self.ds.signalDelays
        )

        # Travel time deltas are the difference between expected and measured arrival times
        # positive deltas are travel times longer than expected
        self.ds["timeDeltas"] = (
            self.ds.measuredTravelTimes - self.ds.expectedTravelTimes
        )

    def filterTravelTimes(self, filterIterations=1):
        """
        Filters the measured travel times.

        This method filters the measured travel times using a predefined filter. It removes outliers based on a pre-defined filter width and filter center of mass. It can iterate over the filtering process until there are no more outliers.

        Args:
            iterateTillConverged (bool): A flag that indicates whether the filter should continue iterating until it no longer finds any outliers.
        """

        filteredSignalTimes = self.ds.measuredTravelTimes.copy()

        for ii in range(filterIterations):
            err = filteredSignalTimes - self.ds.expectedTravelTimes

            filterCondition = np.abs(err - err.median(dim="frame")) < (
                1 + ii / 3
            ) * self.ds.filterStdThreshold * np.abs(err.std(dim="frame"))

            filteredSignalTimes = filteredSignalTimes.where(
                filterCondition
            ).interpolate_na(
                dim="frame",
                method="pchip",  # fill_value="extrapolate"
            )

            if ii == 0:
                self.ds["signalOutlier"] = ~filterCondition
            else:
                self.ds["signalOutlier"] = self.ds["signalOutlier"] + (~filterCondition)

        self.ds["filteredMeasuredTravelTimes"] = filteredSignalTimes

        # collect missing data points
        self.ds["signalDetectionFailure"] = self.ds.measuredTravelTimes.isnull()

        # update time deltas
        self.ds["timeDeltasFiltered"] = (
            self.ds.filteredMeasuredTravelTimes - self.ds.expectedTravelTimes
        )

    ## utility functions
    def to_netcdf(self, filePath) -> None:
        utils.to_netcdf(self.ds.unstack(), filePath)

    @classmethod
    def from_netcdf(cls, filePath):
        cls.ds = utils.from_netcdf(filePath)
        cls.ds = utils.stackPathID(
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

    def stackPathID(
        self,
        dim: str = "pathID",
        stackingDims: list = ["spk", "mic"],
        dropna: bool = True,
    ):
        return utils.stackPathID(
            self.ds, dim=dim, stackingDims=stackingDims, dropna=dropna
        )
