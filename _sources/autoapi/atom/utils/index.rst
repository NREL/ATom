atom.utils
==========

.. py:module:: atom.utils


Functions
---------

.. autoapisummary::

   atom.utils.object_dict
   atom.utils.to_netcdf
   atom.utils.from_netcdf
   atom.utils._check_imported_data
   atom.utils.build_xarray
   atom.utils.save_to_pickle
   atom.utils.load_from_pickle
   atom.utils.describe_dataset
   atom.utils.roll_frame_index
   atom.utils.stackDimension


Module Contents
---------------

.. py:function:: object_dict(obj: dataclasses.dataclass) -> dict

   Convert a dataclass object to a dictionary.

   :param obj: the dataclass object to be converted.
   :type obj: dataclass

   :returns: a dictionary representation of the object.
   :rtype: dict


.. py:function:: to_netcdf(ds, file_path: str) -> None

   Save a Dataset to a NetCDF file.

   :param ds: the dataset to be saved.
   :type ds: xr.Dataset
   :param file_path: the path to save the NetCDF file to.
   :type file_path: str


.. py:function:: from_netcdf(file_path: str) -> xarray.Dataset

   Load a Dataset from a NetCDF file.

   :param file_path: the path to the NetCDF file to be loaded.
   :type file_path: str

   :returns: the loaded dataset.
   :rtype: xr.Dataset


.. py:function:: _check_imported_data(obj: dataclasses.dataclass, ds: xarray.Dataset) -> None

   Check for common names between object attributes and the variables and attributes in a loaded Dataset.

   :param obj: the dataclass object to compare.
   :type obj: dataclass
   :param ds: the loaded dataset to compare.
   :type ds: xr.Dataset


.. py:function:: build_xarray(data_vars: Optional[Dict[str, xarray.DataArray]] = None, coords: Optional[Dict[str, xarray.DataArray]] = None, attrs: Optional[Dict[str, any]] = None, obj: Optional[object] = None, data_vars_exclude: Optional[List[str]] = None) -> xarray.Dataset

   Build an xarray dataset from either an object or a set of dictionaries.

   If `obj` is not None, the dictionaries `data_vars`, `coords`, and `attrs`
   will be constructed from the attributes of the object. Otherwise, the
   dictionaries will be used directly. In either case, if `data_vars_exclude`
   is not None, the listed attributes will be excluded from the `data_vars`
   dictionary.

   :param data_vars: A dictionary of data variables.
   :type data_vars: Dict[str, xr.DataArray], optional
   :param coords: A dictionary of coordinates.
   :type coords: Dict[str, xr.DataArray], optional
   :param attrs: A dictionary of attributes.
   :type attrs: Dict[str, any], optional
   :param obj: An object to use to build the dictionaries.
   :type obj: object, optional
   :param data_vars_exclude: A list of variables to exclude
                             from the `data_vars` dictionary.
   :type data_vars_exclude: List[str], optional

   :returns: An xarray dataset.
   :rtype: xr.Dataset


.. py:function:: save_to_pickle(obj, file_path)

   Saves a dataclass object to a pickle file.

   :param obj: the object to be saved
   :type obj: object
   :param file_path: the file path to save the object to
   :type file_path: str


.. py:function:: load_from_pickle(file_path)

   Loads a dataclass object from a pickle file.

   :param file_path: the file path of the pickle file
   :type file_path: str

   :returns: the loaded dataclass object
   :rtype: object


.. py:function:: describe_dataset(data: Union[xarray.DataArray, xarray.Dataset]) -> pandas.DataFrame

   Compute summary statistics of a given xarray DataArray or Dataset.

   Args:
   - data (xarray.DataArray or xarray.Dataset): xarray data structure to compute summary statistics for

   Returns:
   - pandas.DataFrame: summary statistics of the input xarray data structure


.. py:function:: roll_frame_index(nFrames: int = 5, alignment: str = 'center')

   Generate an array of time delays based on the number of frames, and alignment preference.

   Parameters:
   - nFrames (int): Number of frames.
   - timeDelay (float): Time delay between frames.
   - alignment (str): Alignment preference. Should be 'center', 'forward', or 'backward'.

   Returns:
   - frameIndices (ndarray): Array of time delays.

   Raises:
   - ValueError: If alignment type is not recognized.


.. py:function:: stackDimension(ds, dim: str = 'pathID', stackingDims: list = ['spk', 'mic'], dropna: bool = True)

