# Scratch space, temp
import os
import numpy as np
import pylab
import mahotas as mh
import skimage
from IPython import embed
from skimage.measure import regionprops 
from matplotlib import pyplot as plt
import scipy.misc
import pandas as pd
import os.path
import csv
import watershed # label image by calling watershed.py
import utils # crop cell by calling utils.py
import plot
from PIL import Image
import skimage.io
import scipy
import click
import matplotlib.patches as mpatches
if os.name != 'nt':
  from tsne import bh_sne
from time import time
from matplotlib.ticker import NullFormatter
from sklearn import manifold, datasets
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.switch_backend('agg')
