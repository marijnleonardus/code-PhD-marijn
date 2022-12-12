# author: Deon Janse van Renseburg, Marijn Venderbosch
# december 2022

"""seperate python scripts for devices that we control using Python"""

import pyvisa
import numpy as np
from pylablib.devices import Andor 
import imageio


class ZylaCamera:
    """Controls the Andor Zyla camera used to image the MOT"""

    zyla_camera = 0

    # connect to Andor Zyla camera
    def connect(self):
        if Andor.get_cameras_number_SDK3() != 1:
            print("Andor Camera Not Connected")
        else:
            self.zyla_camera = Andor.AndorSDK3Camera()

            # settings
            self.zyla_camera.set_temperature(0, enable_cooler=True)
            self.zyla_camera.set_attribute_value('SpuriousNoiseFilter', False)
            self.zyla_camera.set_attribute_value('PixelReadoutRate', '100 MHz')
            self.zyla_camera.set_attribute_value('StaticBlemishCorrection', True)
            self.zyla_camera.set_attribute_value('ElectronicShutteringMode', 'Rolling')
            self.zyla_camera.set_attribute_value('SimplePreAmpGainControl', '16-bit (low noise & high well capacity)')
            self.zyla_camera.set_trigger_mode("ext")
            self.zyla_camera.set_attribute_value("ExposureTime", 0.0005)
            self.zyla_camera.set_roi(1037, 1348, 986, 1301, 1, 1)  # horizontal range, vertical range, 1x1 binning

    # Prepare camera to take images
    def start_acquisition(self):
        self.zyla_camera.start_acquisition()

    def end_acquisition(self):
        self.zyla_camera.stop_acquisition()

    # Set the camera in a state to return frames after the TTL is triggered. Images are then saved as 16 bit tif files.
    def save_images(self, path, number_sequences):
        self.start_acquisition()

        run_acquisition_mode = 1
        image_number = 0
        index = 0

        while run_acquisition_mode:

            # acquisition loop
            while image_number < number_sequences:

                # wait for the next available frame
                self.zyla_camera.wait_for_frame() 

                # get the oldest image which hasn't been read yet
                frame = self.zyla_camera.read_oldest_image() 

                # save image
                imageio.imwrite(path + str(image_number) + '.tif', np.uint16(frame))
                image_number = image_number + 1
                index = index + 1

            run_acquisition_mode = 0
        self.end_acquisition()
        
    def disconnect(self):
        self.zyla_camera.close()
