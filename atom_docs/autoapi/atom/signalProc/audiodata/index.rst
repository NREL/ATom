atom.signalProc.audiodata
=========================

.. py:module:: atom.signalProc.audiodata


Classes
-------

.. autoapisummary::

   atom.signalProc.audiodata.AudioData


Module Contents
---------------

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


