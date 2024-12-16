atom.fluctuatingField.modelGrid
===============================

.. py:module:: atom.fluctuatingField.modelGrid


Classes
-------

.. autoapisummary::

   atom.fluctuatingField.modelGrid.ModelGrid


Module Contents
---------------

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


