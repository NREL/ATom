atom.signalProc
===============

.. py:module:: atom.signalProc


Submodules
----------

.. toctree::
   :maxdepth: 1

   /autoapi/atom/signalProc/atarray/index
   /autoapi/atom/signalProc/audiodata/index
   /autoapi/atom/signalProc/auxdata/index
   /autoapi/atom/signalProc/constants/index
   /autoapi/atom/signalProc/traveltimeextractor/index


Classes
-------

.. autoapisummary::

   atom.signalProc.ATArray
   atom.signalProc.AudioData
   atom.signalProc.AuxData
   atom.signalProc.Constants
   atom.signalProc.TravelTimeExtractor


Package Contents
----------------

.. py:class:: ATArray

   A class to hold information about the microphone and speaker arrays and their interactions.

   .. attribute:: nMics

      Number of microphones.

      :type: int

   .. attribute:: micLocations

      Location coordinates of microphones.

      :type: np.ndarray

   .. attribute:: nSpeakers

      Number of speakers.

      :type: int

   .. attribute:: speakerLocations

      Location coordinates of speakers.

      :type: np.ndarray

   .. attribute:: nTravelPaths

      Number of travel paths.

      :type: int

   .. attribute:: hardwareSignalDelays

      Time-delay hardware latencies.

      :type: np.ndarray

   .. attribute:: npsSpeakerDelays

      Time-delay from non-point source speakers.

      :type: np.ndarray


   .. py:attribute:: nMics
      :type:  int


   .. py:attribute:: micLocations
      :type:  numpy.ndarray


   .. py:attribute:: nSpeakers
      :type:  int


   .. py:attribute:: speakerLocations
      :type:  numpy.ndarray


   .. py:attribute:: nTravelPaths
      :type:  int


   .. py:attribute:: hardwareSignalDelays
      :type:  numpy.ndarray


   .. py:attribute:: npsSpeakerDelays
      :type:  numpy.ndarray


   .. py:attribute:: includeCollocated
      :type:  bool
      :value: False



   .. py:attribute:: nIntegralPoints
      :type:  int
      :value: 250



   .. py:method:: __post_init__()


   .. py:method:: calcPathInfo()

      Calculate path vectors, lengths, and orientations based on microphone and speaker locations.

      The method calculates:
      - Path vectors: Vector from speaker to microphone.
      - Path lengths: Distance from speaker to microphone.
      - Path orientations: Angular orientation of the path.



   .. py:method:: setupPathIntegrals()

      Sets up path integral calculations for a given set of speakers and microphones.

      .. rubric:: Notes

      This method does not return any value, but modifies the class instance by adding a new attribute 'self.ds' which is
      an xarray Dataset containing the information about the paths, points for integrations, and Euclidean distances between successive points.

      It expects that the following attributes are pre-defined in the instance:
      'atarray.ds.speakerLocations', 'atarray.ds.micLocations'.

      The total number of points for integration along each path is hard-coded as 250.



   .. py:method:: excludeColocated(dim: str = 'pathID', stackingDims: list = ['spk', 'mic'])


   .. py:method:: plotSpkMicLocations(ax=None, addLabels: bool = True)


   .. py:method:: plotPaths(ax=None, c=None, lw=1, alpha=0.75)


   .. py:method:: to_netcdf(filePath) -> None


   .. py:method:: from_netcdf(filePath)
      :classmethod:



   .. py:method:: stackPathID(dim: str = 'pathID', stackingDims: list = ['spk', 'mic'], dropna: bool = True)


.. py:class:: AudioData

   Class to handle audio data. Includes attributes related to the audio signals
   and methods for loading data, checking data validity, extracting reference signals, and exporting/importing data.

   .. attribute:: samplingFrequency

      The frequency at which the audio data was sampled, default is 20000 Hz.

      :type: float

   .. attribute:: deltaT

      Time increment value, default is 0.05 seconds.

      :type: float

   .. attribute:: recordTimeDuration

      The duration of the audio recording, default is 0.5 seconds.

      :type: float

   .. attribute:: recordTimeDelta

      The time difference between recordings, default is 0.00005 seconds.

      :type: float

   .. attribute:: recordLength

      The length of the audio record, default is 10000.

      :type: int

   .. attribute:: nFrames

      The number of frames in the audio data, default is 120.

      :type: int

   .. attribute:: chirpTimeDuration

      The duration of the chirp signal, default is 0.0058 seconds.

      :type: float

   .. attribute:: chirpRecordLength

      The length of the chirp record, default is 116 samples.

      :type: int

   .. attribute:: chirpCentralFrequency

      The central frequency of the chirp, default is 1200 Hz.

      :type: float

   .. attribute:: chirpBandwidth

      The bandwidth of the chirp, default is 700 Hz.

      :type: float

   .. attribute:: speakerSignalEmissionTime

      Array of speaker signal emission times.

      :type: np.ndarray

   .. attribute:: nMics

      The number of microphones, default is 8.

      :type: int

   .. attribute:: nSpeakers

      The number of speakers, default is 8.

      :type: int

   .. attribute:: windowHalfWidth

      The half-width of the window, default is 0.01 seconds.

      :type: float


   .. py:attribute:: samplingFrequency
      :type:  float
      :value: 20000



   .. py:attribute:: deltaT
      :type:  float
      :value: 0.05



   .. py:attribute:: recordTimeDuration
      :type:  float
      :value: 0.5



   .. py:attribute:: recordTimeDelta
      :type:  float
      :value: 5e-05



   .. py:attribute:: recordLength
      :type:  int
      :value: 10000



   .. py:attribute:: nFrames
      :type:  int
      :value: 120



   .. py:attribute:: chirpTimeDuration
      :type:  float
      :value: 0.0058



   .. py:attribute:: chirpRecordLength
      :type:  int
      :value: 116



   .. py:attribute:: chirpCentralFrequency
      :type:  float
      :value: 1200



   .. py:attribute:: chirpBandwidth
      :type:  float
      :value: 700



   .. py:attribute:: speakerSignalEmissionTime
      :type:  numpy.ndarray


   .. py:attribute:: nMics
      :type:  int
      :value: 8



   .. py:attribute:: nSpeakers
      :type:  int
      :value: 8



   .. py:attribute:: windowHalfWidth
      :type:  float
      :value: 0.01



   .. py:method:: __post_init__()


   .. py:method:: loadData(dataPath, keepSpkData=False)

      Load the main data from the specified path.

      :param dataPath: Path to microphone and speaker data.
      :type dataPath: str
      :param keepSpkData: Flag to indicate whether the full record of speaker data should be kept,
                          or just a reference signal. Default is False.
      :type keepSpkData: bool, optional



   .. py:method:: _checkData(data, nChannels)

      Private method to sanity check data size/shape against config inputs.

      :param data: Data to be checked.
      :type data: numpy.ndarray
      :param nChannels: Number of channels as per config.
      :type nChannels: int



   .. py:method:: getReferenceSignal(spkData)

      Extract a reference signal from one speaker.

      :param spkData: Speaker data from which the reference signal needs to be extracted.
      :type spkData: numpy.ndarray



   .. py:method:: to_netcdf(filePath) -> None


   .. py:method:: from_netcdf(filePath) -> None
      :classmethod:



   .. py:method:: to_pickle(file_path)


   .. py:method:: describe() -> pandas.DataFrame


.. py:class:: AuxData

   This class represents auxiliary data, including sonic anemometer and TH probe data.

   .. attribute:: samplingFrequency

      The sampling frequency of the data.

      :type: float

   .. attribute:: recordTimeDuration

      The time duration of the data in seconds.

      :type: float

   .. attribute:: recordTimeDelta

      The time delta between each sample in seconds.

      :type: float

   .. attribute:: recordLength

      The length of the data.

      :type: int

   .. attribute:: variables

      The variables to include in the data.

      :type: list

   .. attribute:: sonicAnemometerOrientation

      The orientation of the sonic anemometer in degrees.

      :type: float

   .. attribute:: windowType

      The type of window to apply to the velocity signal when calculating the velocity spectrum. Defaults to "hanning".

      :type: str, optional

   .. attribute:: threshold

      The threshold to use when calculating the integral length scale. Defaults to 0.5.

      :type: float, optional


   .. py:attribute:: samplingFrequency
      :type:  float


   .. py:attribute:: recordTimeDuration
      :type:  float


   .. py:attribute:: recordTimeDelta
      :type:  float


   .. py:attribute:: recordLength
      :type:  int


   .. py:attribute:: variables
      :type:  field(default_factory=list)


   .. py:attribute:: sonicAnemometerOrientation
      :type:  float


   .. py:attribute:: windowType
      :type:  str
      :value: 'hann'



   .. py:attribute:: threshold
      :type:  float
      :value: 0.5



   .. py:method:: __post_init__()

      Initialize an `AuxData` object by building a `xarray.Dataset` with default values for the data variables.



   .. py:method:: loadData(auxdatapath, applySonicOrientation=True)

      Load auxiliary data from specified file path.

      :param auxdatapath: path to the auxiliary data file.
      :type auxdatapath: str
      :param applySonicOrientation: flag to indicate whether to reorient the u and v signals to match the orientation of the sonic anemometer in the field. Default is True.
      :type applySonicOrientation: bool, optional

      :returns: None



   .. py:method:: autoCorrelation(variables=['u', 'v', 'uz', 'T', 'H'], lags=None) -> xarray.DataArray

      Calculate the velocity autocorrelation from a turbulent velocity signal.

      :returns: The velocity autocorrelation, with dimensions 'lag' and 'vector_component'.
      :rtype: xarray.DataArray



   .. py:method:: integral_scale(variables=['u', 'v', 'uz', 'T', 'H']) -> None

      Calculate the integral scale from an autocorrelation function.

      :param autocorrelation: The autocorrelation function, with dimensions 'time_lag' and 'vector_component'.
      :type autocorrelation: xr.DataArray
      :param dt: The time lag, with dimension 'time_lag'.
      :type dt: xr.DataArray

      :returns: The integral scale.
      :rtype: float



   .. py:method:: calculateSpectra(nperseg: int = 256, nfft: int = 1024, variables=['u', 'v', 'uz', 'T', 'H']) -> None

      Calculate the velocity spectrum from a turbulent velocity signal using the FFT.

      :returns: The velocity spectrum, with dimensions 'frequency', 'vector_component', and 'spatial_direction'.
      :rtype: xarray.DataArray



   .. py:method:: _rotateHorizontalVelocity(velocityData, angtype='deg')

      Rotate horizontal velocity components reported by the sonic into the reference frame

      :param auxdata: Sonic anemometer and TH probe dta
      :type auxdata: pd.DataFrame
      :param theta: Orientation angle of sonic anemometer
      :type theta: float
      :param angtype: Units of orientation angle. Defaults to "deg".
      :type angtype: str, optional

      :returns: Horizontal components of velocity in reference frame
      :rtype: pd.DataFrame



   .. py:method:: _rotation_matrix(theta, angtype='rad')

      Generate 2D rotation matrix

      :param theta: angle of rotation
      :type theta: float
      :param angtype: units of rotation angle. Defaults to "rad".
      :type angtype: str, optional

      :returns: 2D rotation matrix
      :rtype: np.array



   .. py:method:: _addChannelAttrs() -> None

      add attrs to each xr.dataArray in self.ds



   .. py:method:: to_netcdf(filePath) -> None


   .. py:method:: from_netcdf(filePath)
      :classmethod:



   .. py:method:: to_pickle(file_path)


   .. py:method:: from_pickle(file_path)
      :classmethod:



   .. py:method:: describe() -> pandas.DataFrame


.. py:class:: Constants

   .. py:attribute:: gamma
      :type:  float


   .. py:attribute:: Ra
      :type:  float


.. py:class:: TravelTimeExtractor(configData: dict = {'Note': 'refer to configuration yaml file for correct inputs'}, atarray: xarray.Dataset = None, audiodata: xarray.Dataset = None, auxdata: xarray.Dataset = None, correctSignalDelayEstimate: bool = True)

   A class to perform extraction of travel times from recorded acoustic signals.

   .. attribute:: upsampleFactor

      The factor by which to upsample the data. Default is 10.

      :type: int

   .. attribute:: backgroundFlow

      The background flow model. Can be "homogeneous", "loglaw", "powerlaw", or "FLORIS". Default is "homogeneous".

      :type: str

   .. attribute:: backgroundTemp

      The background temperature model. Default is "homogeneous".

      :type: str

   .. attribute:: filterType

      The type of filter to use. Default is "butter".

      :type: str

   .. attribute:: filterOrder

      The order of the filter. Default is 4.

      :type: int

   .. attribute:: filterScale

      The scale of the filter. Default is 0.5.

      :type: float

   .. attribute:: peakHeight

      The height of the peak. Default is 3.25.

      :type: float

   .. attribute:: peakDistance

      The distance of the peak. Default is 200.

      :type: int

   .. attribute:: filterCOM

      The center of mass for the filter. Default is 99999.0.

      :type: float

   .. attribute:: filterWidth

      The width of the filter. Default is 4.0.

      :type: float

   .. attribute:: delayOffset

      The delay offset. Default is -0.001.

      :type: float

   .. attribute:: atarray

      The acoustic travel time array. Default is None.

      :type: object

   .. attribute:: audiodata

      The audio data. Default is None.

      :type: object

   .. attribute:: auxdata

      The auxiliary data. Default is None.

      :type: object

   .. attribute:: ds

      The dataset containing the attributes of the class.

      :type: xarray.Dataset


   .. py:attribute:: atarray
      :value: None



   .. py:attribute:: audiodata
      :value: None



   .. py:attribute:: auxdata
      :value: None



   .. py:attribute:: correctSignalDelayEstimate
      :value: True



   .. py:attribute:: ds


   .. py:method:: extractTravelTimes()

      Executes the whole travel time extraction process.

      This method performs the entire process of travel time extraction in a sequential manner by calling the necessary internal methods in the right order.



   .. py:method:: signalETAs()

      Calculates the expected signal arrival times based on path and group velocity.

      This method calculates the expected time of arrival of signals at each microphone using the paths and the group velocity. It takes into account the projection of the velocity vector from the sonic onto the travel paths.



   .. py:method:: filterMicData()

      Applies a filter to the microphone data.

      This method applies a selected filter to the recorded microphone data. The type of filter applied depends on the `filterType` attribute. The available options are 'fft' or 'butter'.



   .. py:method:: extractSubSeries(keepSubData=False)

      Extracts subseries of data from the audio signal.

      :param keepSubData: Flag indicating whether to keep the subseries data.
                          If set to False, the subseries data will be discarded.
      :type keepSubData: bool

      :returns: None



   .. py:method:: upsampleMicData()

      Upsample microphone subsamples, associated time vectors, and reference signals.



   .. py:method:: correlateRefSig()

      Correlate microphone data around signal ETAs with reference signal.



   .. py:method:: findPeakCorrelations()

      Detects the peak correlations between the microphone signals and the reference signal.

      This method detects the peaks of correlation between the microphone signals and the reference signal by calculating the cross-correlation between the two signals and identifying the peaks in the correlation result.



   .. py:method:: calculateMeasuredTravelTimes()

      Calculates the measured travel times for each speaker and microphone pair.

      This method calculates the measured travel times for each speaker and microphone pair. It considers the time delay due to the emission time, hardware signal delays and NPS speaker delays. It then compares these with the expected arrival times to determine the travel time deltas.



   .. py:method:: filterTravelTimes(filterIterations=1)

      Filters the measured travel times.

      This method filters the measured travel times using a predefined filter. It removes outliers based on a pre-defined filter width and filter center of mass. It can iterate over the filtering process until there are no more outliers.

      :param iterateTillConverged: A flag that indicates whether the filter should continue iterating until it no longer finds any outliers.
      :type iterateTillConverged: bool



   .. py:method:: to_netcdf(filePath) -> None


   .. py:method:: from_netcdf(filePath)
      :classmethod:



   .. py:method:: to_pickle(file_path)


   .. py:method:: from_pickle(file_path)
      :classmethod:



   .. py:method:: describe()


   .. py:method:: stackPathID(dim: str = 'pathID', stackingDims: list = ['spk', 'mic'], dropna: bool = True)


