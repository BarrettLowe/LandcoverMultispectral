from PIL import Image
import os, numpy as np

import csv

prefix='ref_landcovernet_v1_labels_'

ClassIntMap = {1: 'Water',
               2: 'Artificial Bareground',
               3: 'Natural Bareground',
               4: 'Permanent Snow/Ice',
               5: 'Woody Vegetation',
               6: 'Cultivated Vegetation',
               7: '(Semi) Natural Vegetation',
               8: 'No Data'}

class ImageChip():
    """Class represending an image chip of the dataset
    example usage:
        ic = ImageChip('36MWU','20')
        print(ic.GetBandData(1, "20180107"))
        print(ic.GetCloudProb("20180107"))
        print(ic.GetSceneClass("20180107"))
    """
        
    def __init__(self, tileID:str, chipID: str, prefix='data'):
        self.PATH=os.path.sep.join([prefix,'ref_landcovernet_v1_labels'])
        self.CONVERT_PATH = lambda tileId, chipId, date, band: '{0}_{1}_{2}/source/{1}_{2}_{3}_{4}_10m.tif'.format(self.PATH, tileId, chipId, date, band)

        self.tileId = tileID
        self.chipId = chipID
        self.imgPath = lambda date, band: self.CONVERT_PATH(self.tileId, self.chipId, date, band)

    def GetBandData(self, bandNum:int, date:str):
        bandStr = f'B{bandNum:02d}'
        im = Image.open(self.imgPath(date, bandStr))
        return np.array(im)

    def GetSceneClass(self, date:str):
        im = Image.open(self.imgPath(date, 'SCL'))
        return np.array(im)

    def GetCloudProb(self, date:str):
        im = Image.open(self.imgPath(date, 'CLD'))
        return np.array(im)
        

class DateCSVParser():
    def __init__(self, tileId:str, chipId:str, prefix = 'data'):
        self.tileId = tileId
        self.chipId = chipId
        self._data = []
        filePath = os.path.sep.join([prefix, f'ref_landcovernet_v1_labels_{self.tileId}_{self.chipId}','labels',f'{self.tileId}_{self.chipId}_labeling_dates.csv'])
        with open(filePath) as csvFile:
            reader = csv.reader(csvFile)
            for row in reader:
                self._data.append(row)
        self._data.pop(0) #remove the first one as it's not important
    
    def GetDates(self):
        """Yield next date in a generator expression
        Returns: Generator expression - tuple (index, dateString)
        """
        for x in self._data:
            yield (x[0],x[1])
        
def GetRGB(tileId, chipId, date, prefix='data'):
    chip = ImageChip(tileId, chipId, prefix=prefix)

    # normalize for viewing
    normImg = lambda im: (im.astype(np.float)-im.min())*255.0 / (im.max()-im.min())

    r = chip.GetBandData(4,date)
    g = chip.GetBandData(3,date)
    b = chip.GetBandData(2,date)

    # channels are in floating point due to normalization
    # then convert to 8 bit
    channels = [Image.fromarray(normImg(x)).convert('L') for x in [r,g,b]]

    # merge into 1 rgb image
    image = Image.merge('RGB', channels)
    return image

def CountClassLabels(tileId, chipId, date, prefix='data'):
    chip = ImageChip(tileId, chipId, prefix)

    classes = chip.GetSceneClass(date)
    counts = [(ClassIntMap[x],np.count_nonzero(classes == x)) for x in ClassIntMap.keys()]
    return counts

def FindTilesAndChips(path):
    """Return the tiles and chips that are available in the passed path

    Args:
        path (str): path

    Returns:
        list: tuples with tile,chip
    """
    dirs = os.listdir(path)
    tc = [x[len(prefix):].split('_') for x in dirs]
    return [(x[0],x[1]) for x in tc]

def ParseIdsForTilesAndChips(ids):
    if type(ids) == list:
        ids = map(lambda x:x.split('_'), ids)
        return [(x[-2],x[-1]) for x in ids]
    elif type(ids) == str:
        x = ids.split('_')
        return (x[-2],x[-1])
    else:
        raise TypeError("Input type should be list or string")

def AlreadyHaveData(id, path):
    return ParseIdsForTilesAndChips(id) in FindTilesAndChips(path)
