# author: Deon Janse van Rensburg, Marijn Venderbosch
# cloned from artiq-Rb-setup repository by Deon

import os
from datetime import datetime
from artiq.experiment import *


# Class for handling the saving of experimental data
class SavingExperimentalData:
    """class for handling saving of experimental datda in specified folder"""

    # Create folder on S drive with parent directory named by date and sequence folders numbered
    def create_folder(self):

        save_path = "S:/KAT1/Images/Artiq/SrMachine/" + datetime.today().strftime('%Y-%m-%d/')

        if not os.path.exists(save_path):
            os.makedirs(save_path)
        
        # count number of directories and name folder
        name_folder = str(len(list(os.walk(save_path))))  
        save_directory = os.path.dirname(save_path + name_folder + "/")

        if not os.path.exists(save_directory):
            os.makedirs(save_directory)
        return save_path + name_folder + "/"

    # Write a readme file to the S drive containing the sequence parameters as given in a dictionary
    def write_readme(self, path, param):
        f = open(path + "README.txt", "w")
        for variable, value in param.items():
            f.write(f'{variable} : {value}\n')
        f.close()


class Sr(HasEnvironment):
    """Collection of funcions that can be referenced in the main script""" 

    def build(self):
        self.setattr_device("ttl4")  # camera trigger
        self.setattr_device("urukul0_ch2")  # deflector left/right
        self.setattr_device("urukul0_ch3")  # deflector up/down

    @kernel
    def initialize(self):
        self.ttl4.output()
        self.ttl4.off()  # do not expose camera

        # initialize DDS deflector up/down
        self.urukul0_ch2.cpld.init()
        self.urukul0_ch2.init()

        # write attenuation value
        self.urukul0_ch2.set_att(10*dB)

        # initialize DDS deflector left/right
        self.urukul0_ch3.cpld.init()
        self.urukul0_ch3.init()

        # write atteunation value
        self.urukul0_ch3.set_att(10*dB)

    # take camera image
    @kernel  
    def image_mot(self, exposure_time):
        self.ttl4.pulse(exposure_time*ms)
