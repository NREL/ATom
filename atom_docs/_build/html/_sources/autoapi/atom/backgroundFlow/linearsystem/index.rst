atom.backgroundFlow.linearsystem
================================

.. py:module:: atom.backgroundFlow.linearsystem


Classes
-------

.. autoapisummary::

   atom.backgroundFlow.linearsystem.LinearSystem


Module Contents
---------------

.. py:class:: LinearSystem

   This class is responsible for assembling all the necessary information for the linear system solution as part of the AT process.

   .. math::

       Gf = b


   :math:`G` is the geometry block (which can also include direct measurements).

   :math:`f` is the vector of unknowns :math:`f = [1/c_0, u_0/c_0^2, v0/c_0^2]`

   :math:`b` are observed travel times from travelTimeExtractor

   The background flow if found through the solution:

   .. math::

       f = (G^{-1} * G)^T (G^{-1} * b)

   :math:`G` = [I, 3] (I==nMics*nSpeakers)
   :math:`f` = [3] (see above)
   :math:`f` = I (travel times for a frame)

   .. attribute:: atarray

      The acoustic travel time array.

      :type: object

   .. attribute:: directMeasurements

      The direct measurements for the linear system.

      :type: xr.DataArray

   .. attribute:: regularize

      A flag indicating whether to apply regularization. Default is False.

      :type: bool

   .. attribute:: measuredTravelTime

      The measured travel time for a frame.

      :type: xr.DataArray

   .. attribute:: includeCollocated

      A flag indicating whether to include collocated elements. Default is False.

      :type: bool


   .. py:attribute:: atarray
      :type:  object
      :value: None



   .. py:attribute:: directMeasurements
      :type:  xarray.DataArray
      :value: None



   .. py:attribute:: regularize
      :type:  bool
      :value: False



   .. py:attribute:: measuredTravelTime
      :type:  xarray.DataArray
      :value: None



   .. py:attribute:: constants
      :type:  object
      :value: None



   .. py:method:: __post_init__()

      The geometry block contains all of the path orientation data for the AT array. It should have a size that corresponds to the number of acoustic travel paths, L, the dimensionality of the interrogation area (i.e., 2D or 3D) and a unity vector.



   .. py:method:: executeProcess()

      Executes all methods of the LinearSystem in order. This includes building the process block, collecting the observation block, solving the system, and extracting the bulk values.



   .. py:method:: buildProcessBlock()

      Builds the full matrix block that contains the geometry and auxiliary linear system considerations like direct measurements and model parameters for regularized systems. If direct measurements are included, G becomes a block Toeplitz structure. If regularization is included, G is augmented with model parameters.



   .. py:method:: collectObservationBlock()

      Collects the observations block (b) which includes directly observed travel times. The block may also include direct measurements and model parameters measurements.



   .. py:method:: solve()

      Solves the linear system assembled from the matrix and observation blocks.



   .. py:method:: extractBulkValues()

      Extracts the bulk velocity components and speed of sound from the result of the linear system.



   .. py:method:: to_netcdf(filePath) -> None


   .. py:method:: from_netcdf(filePath)
      :classmethod:



   .. py:method:: to_pickle(file_path)


   .. py:method:: from_pickle(file_path)
      :classmethod:



   .. py:method:: describe()


