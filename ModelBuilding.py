from numpy import dtype
from Helpers import DateCSVParser, ImageChip
# from PIL.Image import Image
import tensorflow_decision_forests as tfdf

# import os
import numpy as np
import pandas as pd
import tensorflow as tf
# import math
import typing

TEN_M_BANDS = [2,3,4,8]


class DFCreator():
    def __init__(self):
        self.selectedData = []
    def GenDF(self, tups: typing.List[typing.Tuple[str,str]], prefix='data', cloudProb = 0, numImages=1):
        """Generate a dataframe using the supplied data

        Args:
            tups (typing.List[typing.Tuple[str,str]]): List of tuples containing (tileId, chipId)
            path (str): path to the dataset
            cloudProb (int): Cloud probability - return values will not have probabilities above this
        """
        ret = pd.DataFrame()
        numSoFar = 0
        for tileId,chipId in tups:
            chip = ImageChip(tileId, chipId, prefix)
            dateCSV = DateCSVParser(tileId, chipId, prefix)
            for ind,date in dateCSV.GetDates():

                # Get the classifications and cloud probabilities for this date
                dateSceneClass = chip.GetSceneClass(date)
                dateCloudProb = chip.GetCloudProb(date)
                tooCloudy = any([x>cloudProb for y in dateCloudProb for x in y])
                if tooCloudy:
                    continue
                self.selectedData.append((tileId,chipId,date))
                dataAtDate = pd.DataFrame()

                # Get all bands at this date
                for band in TEN_M_BANDS:
                    dateBandData = chip.GetBandData(band,date)

                    assert len(dateBandData) > 0
                    assert len(dateSceneClass) == len(dateCloudProb) == len(dateBandData)

                    # Add a column into the dataframe with a title of the band number
                    dataAtDate[f"band{band}"] = dateBandData.flatten().astype(np.float64)

                # add columns for class and cloud probability
                dataAtDate["classification"] = dateSceneClass.flatten()
                dataAtDate["cloudProb"] = dateCloudProb.flatten().astype(np.float64)

                ret = ret.append(dataAtDate, ignore_index=True)
                numSoFar += 1
                if numSoFar>=numImages:
                    break

        return ret


def PredictImage(model, ctd, prefix='/data'):
    chip,tile,date = ctd

    # Get image data into a dataframe
    ImageChip(tile,chip, prefix)
    dfc = DFCreator((tile,chip),prefix)

    # Predict all pixels in the dataframe

    # Reshape predictions into the image