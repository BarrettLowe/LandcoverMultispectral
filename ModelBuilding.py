from numpy import dtype
from Helpers import DateCSVParser, ImageChip, ShowRGB
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
    def GenDF(self, tups: typing.List[typing.Tuple[str,str]], prefix='data'):
        """Generate a dataframe using the supplied data

        Args:
            tups (typing.List[typing.Tuple[int,int]]): List of tuples containing (tileId, chipId)
            path (str): path to the dataset
        """
        ret = pd.DataFrame()
        for tileId,chipId in tups:
            chip = ImageChip(tileId, chipId, prefix)
            dateCSV = DateCSVParser(tileId, chipId, prefix)
            for ind,date in dateCSV.GetDates():

                # Get the classifications and cloud probabilities for this date
                dateSceneClass = chip.GetSceneClass(date)
                dateCloudProb = chip.GetCloudProb(date)
                dataAtDate = pd.DataFrame()

                # Get all bands at this date
                for band in TEN_M_BANDS:
                    dateBandData = chip.GetBandData(band,date)

                    assert len(dateBandData) > 0
                    assert len(dateSceneClass) == len(dateCloudProb) == len(dateBandData)

                    # Add a column into the dataframe with a title of the band number
                    dataAtDate[f"Band{band}"] = dateBandData.flatten().astype(np.float64)

                # add columns for class and cloud probability
                dataAtDate["classification"] = dateSceneClass.flatten()
                dataAtDate["cloudProb"] = dateCloudProb.flatten().astype(np.float64)

                ret = ret.append(dataAtDate, ignore_index=True)
        return ret