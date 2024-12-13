from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Union
import pickle as pk
import xarray as xr
import pandas as pd
import numpy as np
from atom import utils


def object_dict(obj: dataclass) -> dict:
    """
    Convert a dataclass object to a dictionary.

    Args:
        obj (dataclass): the dataclass object to be converted.

    Returns:
        dict: a dictionary representation of the object.
    """
    return {k: str(v) for k, v in asdict(obj).items()}


def to_netcdf(ds, file_path: str) -> None:
    """
    Save a Dataset to a NetCDF file.

    Args:
        ds (xr.Dataset): the dataset to be saved.
        file_path (str): the path to save the NetCDF file to.
    """
    ds.to_netcdf(file_path)


def from_netcdf(file_path: str) -> xr.Dataset:
    """
    Load a Dataset from a NetCDF file.

    Args:
        file_path (str): the path to the NetCDF file to be loaded.

    Returns:
        xr.Dataset: the loaded dataset.
    """
    ds = xr.open_dataset(file_path)
    # _check_imported_data(obj, ds)
    return ds


def _check_imported_data(obj: dataclass, ds: xr.Dataset) -> None:
    """
    Check for common names between object attributes and the variables and attributes in a loaded Dataset.

    Args:
        obj (dataclass): the dataclass object to compare.
        ds (xr.Dataset): the loaded dataset to compare.
    """
    object_keys = list(object_dict(obj).keys())
    ds_keys = list(ds.keys()) + list(ds.attrs.keys())
    assert (
        len(list(set(object_keys).intersection(set(ds_keys)))) > 0
    ), "No keys in common between object attributes and xr.Dataset from NetCDF"


def build_xarray(
    data_vars: Optional[Dict[str, xr.DataArray]] = None,
    coords: Optional[Dict[str, xr.DataArray]] = None,
    attrs: Optional[Dict[str, any]] = None,
    obj: Optional[object] = None,
    data_vars_exclude: Optional[List[str]] = None,
) -> xr.Dataset:
    """
    Build an xarray dataset from either an object or a set of dictionaries.

    If `obj` is not None, the dictionaries `data_vars`, `coords`, and `attrs`
    will be constructed from the attributes of the object. Otherwise, the
    dictionaries will be used directly. In either case, if `data_vars_exclude`
    is not None, the listed attributes will be excluded from the `data_vars`
    dictionary.

    Args:
        data_vars (Dict[str, xr.DataArray], optional): A dictionary of data variables.
        coords (Dict[str, xr.DataArray], optional): A dictionary of coordinates.
        attrs (Dict[str, any], optional): A dictionary of attributes.
        obj (object, optional): An object to use to build the dictionaries.
        data_vars_exclude (List[str], optional): A list of variables to exclude
            from the `data_vars` dictionary.

    Returns:
        xr.Dataset: An xarray dataset.
    """
    if obj is not None:
        # Get object attributes as dict
        item_dict = obj.__dict__
        if data_vars is not None:
            data_vars = {var: item_dict[var] for var in data_vars}
            if data_vars_exclude is not None:
                for var in data_vars_exclude:
                    data_vars.pop(var, None)
        if coords is not None:
            coords = {coord: item_dict[coord] for coord in coords}
        if attrs is not None:
            attrs = {attr: item_dict[attr] for attr in attrs}
    return xr.Dataset(data_vars=data_vars, coords=coords, attrs=attrs)


def save_to_pickle(obj, file_path):
    """
    Saves a dataclass object to a pickle file.

    Args:
        obj (object): the object to be saved
        file_path (str): the file path to save the object to
    """
    with open(file_path, "wb") as f:
        pk.dump(obj, f)


def load_from_pickle(file_path):
    """
    Loads a dataclass object from a pickle file.

    Args:
        file_path (str): the file path of the pickle file

    Returns:
        object: the loaded dataclass object
    """
    with open(file_path, "rb") as f:
        return pk.load(f)


def describe_dataset(data: Union[xr.DataArray, xr.Dataset]) -> pd.DataFrame:
    """
    Compute summary statistics of a given xarray DataArray or Dataset.

    Args:
    - data (xarray.DataArray or xarray.Dataset): xarray data structure to compute summary statistics for

    Returns:
    - pandas.DataFrame: summary statistics of the input xarray data structure
    """
    if isinstance(data, xr.DataArray) | isinstance(data, xr.Dataset):
        data = data.to_dataframe()
    # elif isinstance(data, xr.Dataset):
    #     data = data.to_dataframe().reset_index()#.drop("variable", axis=1)
    else:
        raise ValueError("Input must be xarray DataArray or Dataset.")

    return data.describe()


def roll_frame_index(nFrames: int = 5, alignment: str = "center"):
    """
    Generate an array of time delays based on the number of frames, and alignment preference.

    Parameters:
    - nFrames (int): Number of frames.
    - timeDelay (float): Time delay between frames.
    - alignment (str): Alignment preference. Should be 'center', 'forward', or 'backward'.

    Returns:
    - frameIndices (ndarray): Array of time delays.

    Raises:
    - ValueError: If alignment type is not recognized.
    """

    # create array of time delays of length nFrames
    frameIndices = np.arange(nFrames)
    # shift array of time delays based on alignment preference
    # scale array by the numerical value of self.timeDelay
    if alignment == "center":
        frameIndices = frameIndices - np.ceil((len(frameIndices) - 1) / 2)
        frameIndices = np.sort(frameIndices * -1)
    elif alignment == "forward":
        frameIndices = frameIndices
    elif alignment == "backward":
        frameIndices = np.sort(frameIndices * -1)
    else:
        raise ValueError(
            "Alignment type not recognized. Should be 'center', 'forward', or 'backward'."
        )

    return frameIndices


def stackDimension(
    ds, dim: str = "pathID", stackingDims: list = ["spk", "mic"], dropna: bool = True
):
    ds = ds.stack({dim: stackingDims})
    if dropna:
        # pull out variables that contain the stacked dimension
        varlist = [x for x in ds.data_vars if dim in ds[x].dims]
        subds = ds[varlist]
        subds = subds.where(subds[stackingDims[0]] != subds[stackingDims[1]]).dropna(
            dim=dim
        )
        # pull out variables that do not contain the stacked dimension
        notvarlist = [x for x in ds.data_vars if dim not in ds[x].dims]
        for var in notvarlist:
            subds[var] = ds[var]
        return subds
    else:
        return ds
