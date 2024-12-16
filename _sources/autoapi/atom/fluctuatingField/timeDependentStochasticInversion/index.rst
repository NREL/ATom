atom.fluctuatingField.timeDependentStochasticInversion
======================================================

.. py:module:: atom.fluctuatingField.timeDependentStochasticInversion


Classes
-------

.. autoapisummary::

   atom.fluctuatingField.timeDependentStochasticInversion.TimeDependentStochasticInversion


Module Contents
---------------

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


