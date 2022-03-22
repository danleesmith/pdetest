# Import libraries
import os, math
import numpy as np
import rasterio as rio
import rasterio.merge as riomerge
import mercantile as merc
import fiona

# Import specifics
from shapely.geometry import box, Point, MultiPoint
from torch.utils.data import Dataset

'''
---------------------------------------------
Geographic Utilities
---------------------------------------------
'''

def m_to_dd(m: float, lat: float) -> float:

    '''
    Converts metres into decimal degress (accounts for latitude)
    Credit: @Christofer Ohlsson
    Link: https://stackoverflow.com/questions/25237356/convert-meters-to-decimal-degrees
    '''

    return m / (111.32 * 1000 * math.cos(lat * (math.pi / 180)))

'''
---------------------------------------------
Chips Extractor
---------------------------------------------

For a detailed process explanation see accompanying implementation notebook
'''

def extract_chips(coords_f: str, raster_dir: str, chip_size: int, pix_m: float, obs_years: int, bands_per_year: int) -> list:

    # Get coordinates

    coords = []

    with fiona.open(coords_f, 'r') as c_src:
        for feature in c_src:
            coords += feature['geometry']['coordinates']

    # Get coordinates bounds + chip size buffer

    buffered_coords = []

    for coord in coords:
        buffer_dim = m_to_dd(pix_m * chip_size + 1, coord[1])
        buffer = Point(*coord).buffer(buffer_dim, cap_style=3)
        buffered_coords += list(buffer.exterior.coords)

    buffered_bounds = MultiPoint([Point(*c) for c in buffered_coords]).bounds
    buffered_bbx = box(*buffered_bounds)

    # Get set of all intersecting raster tiles

    r_files = set([r.rstrip('.tif') for r in os.listdir(raster_dir)])

    zoom = int(next(iter(r_files)).split('_')[2])

    tiles = set()

    for coord in list(buffered_bbx.exterior.coords)[:4]:
        tile = merc.tile(coord[0], coord[1], zoom)
        tiles.add(f'{tile.x}_{tile.y}_{tile.z}')

    # Check avaliability of raster files

    if tiles != r_files:

        raise ValueError(f'expecting {tiles} in dir. Found {r_files}')

    
    # Create raster mosaic

    rasters = [os.path.join(raster_dir, f'{t}.tif') for t in tiles]
    mosaic, xform = riomerge.merge(rasters)
    mosaic_bounds = rio.transform.array_bounds(mosaic.shape[1], mosaic.shape[2], xform)
    
    # Get coordinates as pixel coordinates

    xs = np.array([c[0] for c in coords])
    ys = np.array([c[1] for c in coords])
    rows, cols = rio.transform.rowcol(xform, xs, ys, op=math.floor)

    # Extract chips and output

    chips = []

    for i in range(len(rows)):
        for year in range(obs_years):
            start = year * bands_per_year
            end = start + bands_per_year
            r_clip = mosaic[start : end, (rows[i]-chip_size) : (rows[i]+chip_size+1), (cols[i]-chip_size) : (cols[i]+chip_size+1)]
            chips.append(r_clip)

    return chips

'''
---------------------------------------------
Dataset Utilities
---------------------------------------------
'''

class ChipsDataset(Dataset):

    def __init__(self, data, coords):
        self.data = data # Expecting chip array: [[location][year][chip]]
        self.coords = coords # Expecting a shapely point

    def __len__(self):
        return(len(self.data))

    def __getitem__(self, index):
        x = self.data[index]
        y = self.coords[index]
        return x, y