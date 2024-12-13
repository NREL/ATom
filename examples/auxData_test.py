import hydra
from hydra.utils import instantiate
import os

import pandas as pd
import numpy as np

import pickle as pk
from copy import deepcopy


@hydra.main(
    version_base=None,
    config_path="conf",
    config_name="configs",
)
def my_app(cfg) -> None:
    print(cfg.auxdata.sonicAnemometerOrientation)

    dataDir = "../data/Data_collection_20190815/"
    files = os.listdir(dataDir)
    files.sort()
    auxFiles = [dataDir + x for x in files if "AuxData" in x]

    for ii, auxDataPath in enumerate(auxFiles):
        ### Auxiliary data
        auxdata = instantiate(cfg.auxdata)
        auxdata.loadData(auxDataPath)

        if ii == 0:
            auxdf = deepcopy(auxdata.auxdata)
        else:
            auxdf = pd.concat([auxdf, deepcopy(auxdata.auxdata)], ignore_index=True)

    # index
    auxdf.index = pd.TimedeltaIndex(data=np.arange(len(auxdf)) / 20, unit="S")
    auxdf = auxdf.resample("0.5S").mean()

    # Store the sonic anemometer dataframe
    with open(
        "../data/processedData/auxOutput_{}.pk".format(
            auxdata.sonicAnemometerOrientation
        ),
        "wb",
    ) as file:
        # dump information to that file
        pk.dump(auxdf, file)
    file.close()

    return


if __name__ == "__main__":
    my_app()
