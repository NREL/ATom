atom.fluctuatingField.covarinceMatix
====================================

.. py:module:: atom.fluctuatingField.covarinceMatix


Classes
-------

.. autoapisummary::

   atom.fluctuatingField.covarinceMatix.CovarianceMatrices


Functions
---------

.. autoapisummary::

   atom.fluctuatingField.covarinceMatix._blockToeplitz
   atom.fluctuatingField.covarinceMatix._defineStencil
   atom.fluctuatingField.covarinceMatix._distributeProcesses


Module Contents
---------------

.. py:class:: CovarianceMatrices(configDict: dict, modelGrid: xarray.Dataset, atarray: xarray.Dataset, linearSystem: xarray.Dataset)

   This class is responsible for calculating the covariance matrices used in spatial covariance function calculations.

   .. attribute:: modelGrid

      Contains the model grid data.

      :type: xr.Dataset

   .. attribute:: atarray

      Contains acoustic tomography array data.

      :type: xr.Dataset

   .. attribute:: linearSystem

      Contains the bulk linear system data.

      :type: xr.Dataset

   .. attribute:: sigmaT

      Standard deviation of temperature. Default is 0.14.

      :type: float

   .. attribute:: lT

      Length scale of temperature. Default is 15.0.

      :type: float

   .. attribute:: sigmaU

      Standard deviation of u-component of wind. Default is 0.72.

      :type: float

   .. attribute:: sigmaV

      Standard deviation of v-component of wind. Default is 0.42.

      :type: float

   .. attribute:: l

      Length scale of wind. Default is 15.0.

      :type: float

   .. attribute:: nFrames

      Number of *additional* frames to consider in covariance matrices and TDSI (the N in N+1 frames). Default is 0.

      :type: int

   .. attribute:: alignment

      Alignment of the stencil selecting frames. Default is 'center'.

      :type: str


   .. py:attribute:: modelGrid


   .. py:attribute:: atarray


   .. py:attribute:: ds


   .. py:method:: covarianceFunction(modelGrid: xarray.Dataset = None, sigma: float = None, l: float = None, component: str = None)

      Calculates the spatial covariance function given a model grid and specific parameters. When timeDelay ~= 0, the second spatial coordinate in the covariance function r' is shifted by the advection velocity and time delay  r' + U delta t, assuming Taylor's frozen field hypothesis.

      :param modelGrid: The dataset containing the model grid with variables "x" and "y".
                        Default is None.
      :type modelGrid: xr.Dataset, optional
      :param sigma: Standard deviation of the process. For "uv" correlation, a length 2 array
                    representing standard deviation for x and y is expected. Default is None.
      :type sigma: float, optional
      :param l: Length scale of the process. Default is None.
      :type l: float, optional
      :param component: Specifies the type of component to be used for covariance calculation.
                        Valid inputs are "TT", "uu", "vv", and "uv". Default is None.
      :type component: str, optional
      :param timeDelay: time delay between frames in seconds. Default is 0.0.
      :type timeDelay: float, optional
      :param advectionVelocity: advection velocity
      :type advectionVelocity: np.ndarray, optional

      :returns: The calculated covariance matrix for the specified component.
      :rtype: numpy.ndarray

      :raises AssertionError: If the component is "uv" and sigma is not a length 2 array.

      .. rubric:: Notes

      If the component is None, the function will prompt the user to specify a component.



   .. py:method:: calculateCovariances() -> None

      Calculates and assigns the covariance function values to the current dataset object.

      .. rubric:: Notes

      This method does not return any value, but directly modifies the dataset within the object instance.
      It expects that the following attributes are pre-defined in the dataset: modelGrid, sigmaT, lT, sigmaU,
      sigmaV, and l. The 'modelXY' coordinate is created by stacking 'x' and 'y' from the dataset.



   .. py:method:: modelDataCovarianceMatrix() -> None

      Calculates the model-data covariance matrix and adds it to the instance's dataset.

      .. rubric:: Notes

      This method does not return any value, but modifies the 'ds' attribute of the class instance by
      adding the 'Bmd' data array.
      It expects that the following attributes are pre-defined in the instance: 'pathData', 'ds',
      'modelGrid', 'nModelPointsX', 'nModelPointsY', 'atarray.ds.pathOrientation'.
      The 'ds' attribute is an xarray Dataset instance and 'atarray' is an attribute of the class where
      this method is defined.



   .. py:method:: dataDataCovarianceMatrix() -> xarray.DataArray

      Calculates the data-data covariance matrix and adds it to the instance's dataset.

      .. rubric:: Notes

      This method does not return any value, but modifies the 'ds' attribute of the class instance by
      adding the 'Bdd' data array.
      It expects that the following attributes are pre-defined in the instance: 'pathData', 'ds',
      'modelGrid', 'nModelPointsX', 'nModelPointsY', 'ds.pathOrientation'.
      The 'ds' attribute is an xarray Dataset instance and 'modelGrid' is an attribute of the class where
      this method is defined.



   .. py:method:: assembleTDSICovarianceMatrices()

      Assembles the TDSI (Time-Dependent Stochastic Inversion) covariance matrices.

      :param nFrames: The number of frames to consider. If not provided, it uses the value of self.nFrames.
      :type nFrames: int, optional
      :param alignment: The alignment preference for time delays. Accepts 'center', 'forward', or 'backward'.
                        If not provided, it uses the value of self.alignment.
      :type alignment: str, optional

      :raises ValueError: If the alignment type is not recognized.

      :returns: None



   .. py:method:: _checkDomainLimits()

      Checks if the domain limits are within the domain of the data.

      Args:
      modelx (numpy.ndarray): x-coordinates of the data.
      modely (numpy.ndarray): y-coordinates of the data.
      xlim (list): x-limits of the domain.
      ylim (list): y-limits of the domain.

      Returns:
      bool: True if the domain limits are within the domain of the data.



   .. py:method:: makeDummyCoords()

      make dummy coordinates for the model data so that it can be stacked, unstacked, and subselected without errors.



   .. py:method:: to_netcdf(filePath) -> None


   .. py:method:: from_netcdf(filePath) -> None
      :classmethod:



   .. py:method:: to_pickle(file_path)


   .. py:method:: from_pickle(file_path)
      :classmethod:



   .. py:method:: describe()


.. py:function:: _blockToeplitz(matList: list)

   This function takes a list of matrices and constructs a block Toeplitz matrix.

   #TODO Theory suggest that the lower triangular region of the block Toeplitz matrix should actually be rotated 180 degrees A = np.flip(A, axis=[0,1]).

   :param matList: A list of 2D numpy arrays representing the matrices to be used in the block Toeplitz matrix.
   :type matList: list

   :returns: A block Toeplitz matrix constructed from the input list of matrices.
   :rtype: numpy.ndarray


.. py:function:: _defineStencil(nFrames: int = 0, totalFrames: int = 10, alignment: str = 'center', advectionScheme: str = 'stationary')

   Defines a stencil array for selecting frames from a sequence.

   The stencil array contains indices for selecting frames centered around
   a given frame, with configurable number of frames and alignment. Used
   to construct covariance matrices from frame sequences.



.. py:function:: _distributeProcesses(function, arguments)

   placeholder for now
   #TODO someday...


