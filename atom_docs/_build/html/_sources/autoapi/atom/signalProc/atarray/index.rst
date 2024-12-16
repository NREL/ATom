atom.signalProc.atarray
=======================

.. py:module:: atom.signalProc.atarray


Attributes
----------

.. autoapisummary::

   atom.signalProc.atarray.logger


Classes
-------

.. autoapisummary::

   atom.signalProc.atarray.ATArray


Module Contents
---------------

.. py:data:: logger

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


