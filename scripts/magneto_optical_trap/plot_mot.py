import matplotlib.pyplot as plt
import sys
import os

# add local modules
script_dir = os.path.dirname(os.path.abspath(__file__))
lib_dir = os.path.abspath(os.path.join(script_dir, '../../lib'))
if lib_dir not in sys.path:
    sys.path.append(lib_dir)
from setup_paths import add_local_paths
add_local_paths(__file__, ['../../modules', '../../utils'])

from image_analysis_class import MOTPlot
from camera_image_class import CameraImage
from plot_utils import Plotting
from units import nm, um

# constants
wavelength = 461*nm  # in meters
px_size = 3.45*um  # in meters
binning = 4  # binning factor
magnification = 0.85  # magnification factor

path = r'//physstor/cqt-t/KAT1/Marijn/thesis_measurements/mot/sf_time_of_flight/second try/scan091510/'
#file_name = '0007fluorescence.tif' # t=0ms
file_name = '0017fluorescence.tif' # t=10ms
x0=258 # px
y0=200
w=70
h=120

ImageObject = CameraImage()
fluor_img = ImageObject._load_image(path + file_name)   
fluor_img = ImageObject.crop_image_around_point(fluor_img, x0=x0, y0=y0, w=w, h=h)

FluorescencePlot = MOTPlot(fluor_img, magnification, px_size, binning)
FluorescencePlot.plot_withscalebar(scalebar_length_mm=0.5)
Plot = Plotting('output')
Plot.savefig('fluor_red_10ms.png')
plt.show()