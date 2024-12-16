atom.fluctuatingField.pathSetup
===============================

.. py:module:: atom.fluctuatingField.pathSetup


Classes
-------

.. autoapisummary::

   atom.fluctuatingField.pathSetup.PathSetup


Module Contents
---------------

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


