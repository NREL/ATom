atom.fluctuatingField
=====================

.. py:module:: atom.fluctuatingField


Submodules
----------

.. toctree::
   :maxdepth: 1

   /autoapi/atom/fluctuatingField/covarinceMatix/index
   /autoapi/atom/fluctuatingField/modelGrid/index
   /autoapi/atom/fluctuatingField/pathSetup/index
   /autoapi/atom/fluctuatingField/timeDependentStochasticInversion/index


Classes
-------

.. autoapisummary::

   atom.fluctuatingField.ModelGrid
   atom.fluctuatingField.PathSetup
   atom.fluctuatingField.CovarianceMatrices
   atom.fluctuatingField.TimeDependentStochasticInversion


Package Contents
----------------

.. py:class:: ModelGrid(nModelPointsX, nModelPointsY, modelLimsX, modelLimsY)

   A class that represents a model grid for TDSI solution.

   .. attribute:: nModelPointsX

      Number of points in the x-direction.

      :type: int

   .. attribute:: nModelPointsY

      Number of points in the y-direction.

      :type: int

   .. attribute:: modelLimsX

      Lower and upper bounds for the x-direction.

      :type: Tuple[float, float]

   .. attribute:: modelLimsY

      Lower and upper bounds for the y-direction.

      :type: Tuple[float, float]

   Methods:
   buildModelGrid(): Generates a model grid using the given parameters.
   getDataset(): Returns the xarray Dataset representing the model grid.


   .. py:attribute:: nModelPointsX


   .. py:attribute:: nModelPointsY


   .. py:attribute:: modelLimsX


   .. py:attribute:: modelLimsY


   .. py:attribute:: ds


   .. py:method:: buildModelGrid() -> None

      Generates a model grid using the given parameters.

      This method creates a model grid as a 2D array, where 'x' and 'y' values are equally spaced based on the input parameters.
      The grid is then saved as an xarray DataArray in the 'modelGrid' attribute of the instance.

      Note: This method does not return any value, but directly modifies the 'modelGrid' attribute of the class instance.



   .. py:method:: getDataset()

      Returns the xarray Dataset representing the model grid.



   .. py:method:: to_netcdf(filePath) -> None


   .. py:method:: from_netcdf(filePath) -> None
      :classmethod:



   .. py:method:: to_pickle(file_path)


   .. py:method:: from_pickle(file_path)
      :classmethod:



   .. py:method:: describe()


.. py:class:: PathSetup(atarray)

   This class is responsible for setting up path integral calculations for a given set of speakers and microphones.

   .. attribute:: atarray

      Contains the speakers and microphones data.

      :type: object

   .. attribute:: ds

      An xarray Dataset to store the calculations. Initially an empty dataset.

      :type: xr.Dataset


   .. py:attribute:: atarray


   .. py:attribute:: ds


   .. py:method:: setupPathIntegrals()

      Sets up path integral calculations for a given set of speakers and microphones.

      .. rubric:: Notes

      This method does not return any value, but modifies the class instance by adding a new attribute 'self.ds' which is
      an xarray Dataset containing the information about the paths, points for integrations, and Euclidean distances between successive points.

      It expects that the following attributes are pre-defined in the instance:
      'atarray.ds.speakerLocations', 'atarray.ds.micLocations'.

      The total number of points for integration along each path is hard-coded as 250.



   .. py:method:: getDataset()

      Returns the xarray Dataset representing the path integrals.



   .. py:method:: to_netcdf(filePath) -> None


   .. py:method:: from_netcdf(filePath) -> None
      :classmethod:



   .. py:method:: to_pickle(file_path)


   .. py:method:: from_pickle(file_path)
      :classmethod:



   .. py:method:: describe()


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


.. py:class:: TimeDependentStochasticInversion(modelGrid: xarray.Dataset, atarray: xarray.Dataset, covarMatrices: xarray.Dataset, bulkFlowData: xarray.Dataset, stencil: Iterable[int] = [-1, 0, 1], frameSets: Iterable[int] = [-2, -1, 0, 1, 2], retrieval: str = 'single')

   Instantiates the TimeDependentStochasticInversion class with the provided datasets.

   :param modelGrid: The dataset containing the model grid.
   :type modelGrid: xr.Dataset
   :param atarray: The dataset containing the path data.
   :type atarray: xr.Dataset
   :param covarMatrices: The dataset containing the covariance matrices.
   :type covarMatrices: xr.Dataset


   .. py:attribute:: modelGrid


   .. py:attribute:: atarray


   .. py:attribute:: covarMatrices


   .. py:attribute:: bulkFlowData


   .. py:attribute:: retrieval
      :value: 'single'



   .. py:attribute:: stencil


   .. py:attribute:: frameSets


   .. py:method:: optimalStochasticInverseOperator()

      Calculates the optimal stochastic inverse operator and adds it to the instance's dataset.

      This method computes the matrix product of the reshaped model-data covariance matrix and the inverse of the data-data covariance matrix. The resulting matrix 'A', representing the optimal stochastic inverse operator, is then reshaped and stored in the instance's dataset as 'A'.

      .. rubric:: Notes

      This method does not return any value, but modifies the 'ds' attribute of the class instance by adding the 'A' data array.
      It expects that the following attributes are pre-defined in the instance: 'ds', 'ds.Rmd', 'ds.Rdd', 'nModelPointsX', 'nModelPointsY', and 'atarray'.
      The 'ds' attribute is an xarray Dataset instance.



   .. py:method:: assembleDataVector()

              Assembles the data vector `d`.

              Notes:
                  This method does not return any value, but modifies the 'ds' attribute of the class instance by adding the 'd' data array.
                  The data vector `d` is assembled as:

                  .. math::

                      \mathbf{d} = \left[ \mathbf{d}(t - N    au)~;~ \mathbf{d}(t - N         au+     au)~;~\hdots ~;~ \mathbf{d}(t)
      ight]





   .. py:method:: repackDataVector()


   .. py:method:: getFrameSets()


   .. py:method:: calculateFluctuatingFields()

      Calculates the fluctuating fields and adds it to the instance's dataset.

      .. rubric:: Notes

      This method does not return any value, but modifies the 'ds' attribute of the class instance by adding the 'm' data array.
      The fluctuating fields `m` are calculated as:

      .. math::

          m = \mathbf{A} \mathbf{d}

      where `m` contains all of the fluctuating field data :math:`m = [T'(r,t),u'(r,t),v'(r,t)]`.



   .. py:method:: to_netcdf(filePath) -> None


   .. py:method:: from_netcdf(filePath) -> None
      :classmethod:



   .. py:method:: to_pickle(file_path)


   .. py:method:: from_pickle(file_path)
      :classmethod:



   .. py:method:: describe()


