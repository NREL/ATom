import hydra
from hydra.utils import instantiate
# from hydra import compose, initialize_config_dir, compose
# import os

import numpy as np
import matplotlib.pyplot as plt

# import pickle as pk
# from time import time
import atom


# # Global configuration import for CLI debugging
# def getConfig():
#     configPath = os.path.abspath('conf')
#     initialize_config_dir(version_base=None, config_dir=configPath)
#     cfg=compose(config_name="configs.yaml")
#     return cfg

@hydra.main(
    version_base=None,
    config_path="conf",
    config_name="configs",
)
def my_app(cfg) -> None:

    ### Array data
    atarray = instantiate(cfg.atarray)

    ### Microphone data
    audiodata = instantiate(cfg.audiodata)
    mainDataPath = "/Users/nhamilt2/Documents/ATom/data/Data_collection_20190815/20190815123732_AcouTomMainData.txt"
    audiodata.loadData(mainDataPath)

    ### Auxiliary data
    auxdata = instantiate(cfg.auxdata)
    auxDataPath = "/Users/nhamilt2/Documents/ATom/data/Data_collection_20190815/20190815123732_AcouTomAuxData.txt"
    auxdata.loadData(auxDataPath)

    ## Constants
    constants = instantiate(cfg.constants)

    ## Background flow estimator
    # flow model parameters from AuxData
    # Uref, Zref, etc.
    # FLORIS -> update sampling for domain...
    # Temperature from Rogers and Finn
    # Domain and resolution from array object
    print('cfg object, external values')
    print(cfg.traveltimeextractor)

    ## TravelTimeExtractor
    ttextractor = atom.TravelTimeExtractor(  # instantiate(
        **cfg.traveltimeextractor,
        atarray=atarray,
        audiodata=audiodata,
        auxdata=auxdata,
        constants=constants,
    )
    ttextractor.extractTravelTimes()#upsampleData=True)
    # travelTimes = ttextractor.detectedSignalTimes
    # open a file, where you ant to store the data
    # with open("tra`````
    # fig.savefig()




    ## feed updated background flow back into signalETAs and ttextractor?
    # How many times do we iterate?
    # What's a meaningful convergence criteria?
    # No need to re-detect signal arrival times, just deltas

    ## Fluctuating field reconstruction
    # TDSI
    # EBF
    # UKF
    # others...?
    return


if __name__ == "__main__":
    my_app()


##### Not in use any more.
## estimate signal arrival times from sonic data
# starttime = time()
# ttextractor.extractTravelTimes()
# coarseTT = ttextractor.detectedSignalTimes
# endtime = time()
# print("coarse process:", endtime - starttime)

# starttime = time()
# ttextractor.extractTravelTimes(upsampleData=True)
# fineTT = ttextractor.detectedSignalTimes
# endtime = time()
# print("fine process:", endtime - starttime)

# delta = coarseTT - fineTT
# ave = delta.reshape([-1, 120]).mean(axis=-1)
# std = delta.reshape([-1, 120]).std(axis=-1)
# pathNum = np.arange(delta.reshape([-1, 120]).shape[0])
# fig, ax = plt.subplots()
# plt.plot(ave, label="ave diff")
# plt.fill_between(
#     pathNum,
#     ave - std,
#     ave + std,
#     facecolor="C3",
#     alpha=0.5,
#     zorder=-1,
#     label="std diff",
# )
# plt.title("Difference in Travel time, coarse v. upsampled")
# plt.legend()
# plt.xlabel("Path number")
# plt.ylabel("Time [s]")
# plt.show()

# # plots for degug

# showMicsandETAwindows(ttextractor)
# showTravelTimes(ttextractor)

# plt.show()


def showMicsandETAwindows(ttextractor, mic=0, skip=8):
    ##### Compare recorded/filtered signals to ETAs
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(
        np.arange(0, 0.5000, 1 / ttextractor.audiodata.samplingFrequency),
        ttextractor.micDataFiltered[:, mic, ::skip],
    )
    tmp = ttextractor.expectedSignalArrivalTimes.mean(axis=-1)[:, mic]
    for ii in range(8):
        plt.axvspan(
            tmp[ii] - 0.005, tmp[ii] + 0.01, facecolor="C1", alpha=0.2, zorder=-10
        )
    # plt.show()


def showTravelTimes(ttextractor):
    tmp = ttextractor.detectedSignalTimes.reshape(-1, 120)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(tmp.mean(axis=-1), label="average travel time")
    ax.fill_between(
        np.arange(tmp.shape[0]),
        tmp.mean(axis=-1) - tmp.std(axis=-1),
        tmp.mean(axis=-1) + tmp.std(axis=-1),
        facecolor="C3",
        # alpha=,
        zorder=-2,
        label="std. dev. travel time",
    )
    plt.xlabel("Path Number")
    plt.legend()
    # plt.show()
    