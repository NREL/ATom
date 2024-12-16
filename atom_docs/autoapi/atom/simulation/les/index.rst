atom.simulation.les
===================

.. py:module:: atom.simulation.les


Classes
-------

.. autoapisummary::

   atom.simulation.les.LESData


Module Contents
---------------

.. py:class:: LESData(loadPath=None, atarray: xarray.Dataset = None, constants: xarray.Dataset = None)

   .. py:attribute:: loadPath
      :value: None



   .. py:attribute:: atarray
      :value: None



   .. py:attribute:: constants
      :value: None



   .. py:method:: load_data()

      #TODO add validator



   .. py:method:: get_bulk_values()


   .. py:method:: setup_interpolators()


   .. py:method:: interpolate_field(times=None)

      The method interpolates LES data at given points.
      - times: times over which to interpolate velocity and temperature fields.
          Defaults to None, in which case, all times from the LES data are used.
      :return: Interpolated values at the given points.



   .. py:method:: calculate_travel_time()

      Calculate the travel time using Equation 5.

      :returns: Calculated travel time for the path including measurement error.
      :rtype: float



   .. py:method:: to_netcdf(filePath) -> None


   .. py:method:: from_netcdf(filePath) -> None
      :classmethod:



   .. py:method:: to_pickle(file_path)


   .. py:method:: from_pickle(file_path)
      :classmethod:



   .. py:method:: describe()


