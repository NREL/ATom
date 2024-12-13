import hydra
from hydra.utils import instantiate
import os
import atom
import pandas as pd
import numpy as np
import datetime


@hydra.main(
    version_base=None,
    config_path="conf",
    config_name="configs",
)
def my_app(cfg) -> None:

    dataDir = "../data/Data_collection_20190815/"
    files = os.listdir(dataDir)
    files.sort()
    mainFiles = [dataDir + x for x in files if "MainData" in x][:-1]
    auxFiles = [dataDir + x for x in files if "AuxData" in x][:-1]

    ### Array data
    atarray = instantiate(cfg.atarray)
    ## Constants
    constants = instantiate(cfg.constants)

    for ii, (mainDataPath, auxDataPath) in enumerate(zip(mainFiles, auxFiles)):

        date = mainDataPath.split("/")[-1].split("_")[0]
        print(date)
        dtInd = pd.TimedeltaIndex(np.arange(120) * 0.5, unit="S")
        strdate = datetime.datetime.strptime(date, "%Y%m%d%H%M%S")
        dtInd = dtInd + strdate

        ### Microphone data
        audiodata = instantiate(cfg.audiodata)
        audiodata.loadData(mainDataPath)

        ### Auxiliary data
        auxdata = instantiate(cfg.auxdata)
        auxdata.loadData(auxDataPath)
        auxdata.ds = auxdata.ds.assign_coords(
            time=auxdata.ds.time.to_pandas() + strdate
        )
        auxdata.to_netcdf(f"processedData/auxdata_{date}.nc")

        ## TravelTimeExtractor
        ttextractor = atom.TravelTimeExtractor(  # instantiate(
            **cfg.traveltimeextractor,
            atarray=atarray,
            audiodata=audiodata,
            auxdata=auxdata,
            constants=constants,
        )
        ttextractor.extractTravelTimes()
        # ttextractor.to_netcdf('file.nc')
        
        # ttextractor = atom.TravelTimeExtractor.from_netcdf('file.nc')


        ## Linear system solver
        ls = atom.LinearSystem(
            atarray=atarray,
            measuredTravelTime=ttextractor.ds.filteredMeasuredTravelTimes,
        )
        ls.doIt()

        ls.ds = ls.ds.assign_coords(frame=ls.ds.frame + ii * 120)
        ls.to_netcdf(f"processedData/arrayOutput_{date}.nc")

        #TDSI process

    return


if __name__ == "__main__":
    my_app()
