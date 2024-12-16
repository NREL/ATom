atom.signalProc.auxdata
=======================

.. py:module:: atom.signalProc.auxdata


Classes
-------

.. autoapisummary::

   atom.signalProc.auxdata.AuxData


Module Contents
---------------

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


