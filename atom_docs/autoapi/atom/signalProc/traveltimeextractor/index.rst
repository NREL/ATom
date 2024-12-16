atom.signalProc.traveltimeextractor
===================================

.. py:module:: atom.signalProc.traveltimeextractor


Classes
-------

.. autoapisummary::

   atom.signalProc.traveltimeextractor.TravelTimeExtractor


Module Contents
---------------

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


