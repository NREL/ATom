atom.backgroundFlow.flowModel
=============================

.. py:module:: atom.backgroundFlow.flowModel


Classes
-------

.. autoapisummary::

   atom.backgroundFlow.flowModel.FlowModel
   atom.backgroundFlow.flowModel.homogeneous
   atom.backgroundFlow.flowModel.logLaw
   atom.backgroundFlow.flowModel.powerLaw


Module Contents
---------------

.. py:class:: FlowModel

   The background flow object describes the background atmospheric velocity and temperature fields around which spatial fluctuations are to be calculated.


.. py:class:: homogeneous

   Bases: :py:obj:`FlowModel`


   Constant homogeneous background flow. Spatial average values of u, v, and T should be supplied from the AuxData (sonic anemometer). If no average values for u, v, or T are supplied, defaults to None.


   .. py:method:: __post_init__(meanU, meanV, meanT)


   .. py:method:: rotateVel(flowDir: float = 0.0) -> None

      Rotate velocity field to match measured flow direction.



.. py:class:: logLaw

   Bases: :py:obj:`FlowModel`


   The log wind profile is a semi-empirical relationship commonly used to describe the vertical distribution of horizontal mean wind speeds within the lowest portion of the planetary boundary layer.
   reference: https://en.wikipedia.org/wiki/Log_wind_profile


   .. py:attribute:: frictionVel
      :type:  float


   .. py:attribute:: kappa
      :type:  float


   .. py:attribute:: roughnessHeight
      :type:  float


   .. py:attribute:: displacementHeight
      :type:  float


   .. py:method:: __post_init__()


   .. py:method:: logLawProfile(zVec)


.. py:class:: powerLaw

   Bases: :py:obj:`FlowModel`


   Calculates velocity following a power law with a reference velocity at a reference height and a shear exponent.
   reference: https://en.wikipedia.org/wiki/Wind_profile_power_law


   .. py:attribute:: uRef
      :type:  float


   .. py:attribute:: zRef
      :type:  float


   .. py:attribute:: shearExponent
      :type:  float


   .. py:method:: __post_init__()


   .. py:method:: powerLawProfile(zVec)


