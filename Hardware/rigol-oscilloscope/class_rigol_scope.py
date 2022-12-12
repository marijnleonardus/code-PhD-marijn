# -*- coding: utf-8 -*-
"""
Created on Fri Jul 29 11:02:33 2022

@author: Rik van Herk
"""

import pyvisa as pv
import numpy as np


class RigolScope:
        
    def __init__(self, ip_address):
        """make connection to oscilloscope"""
        
        self.rm = pv.ResourceManager()
        self.ip = ip_address
        self.device = self.rm.open_resource(f"TCPIP::{self.ip}::INSTR")

    def set_mode(self, mode):
        """set mode to run/stop/single"""
        
        self.device.write(mode.lower())

    def trigger(self):
        """force scope trigger"""
        
        self.device.write(':TFOR')

    def set_averages(self, n):
        """set aquire averages"""
        
        if n == 1:
            self.device.write('ACQ:TYPE NORM')
        elif 1 <= n <= 1024:
            self.device.write('ACQ:TYPE AVER')
            self.device.write('ACQ:AVER '+str(n))
        else:
            self.device.write('ACQ:TYPE NORM')
            print("Averages has to be between 1 and 1024, averages set to 1")

    def set_coupling(self, channel, coupling):
        """set AC or DC coupling"""
        
        self.device.write(f":CHAN{channel}:COUP{coupling}")

    def get_data(self, channel=1):
        """returns x(s),y(V) data currently on the scope screen on given channel"""
        
        timeoffset = float(self.device.query(":TIM:OFFS?"))
        timescale = float(self.device.query(":TIM:SCAL?"))
        voltscale = float(self.device.query(':CHAN'+str(channel)+':SCAL?'))
        voltoffset = float(self.device.query(":CHAN"+str(channel)+":OFFS?"))
        self.device.write(":WAV:POIN:MODE RAW")
        self.device.write(":WAV:SOUR CHAN"+str(channel))
        self.device.write(":WAV:DATA?")
        rawdata = self.device.read_raw()[11:]
        #  if rawdata.decode("utf-8")=='\n':  #fix for no data
        #      return [],[]
        data = np.frombuffer(rawdata, 'B') * -1 + 255
        data = (data - 130.0 - voltoffset/voltscale*25) / 25 * voltscale
        t = np.linspace(timeoffset - 6 * timescale, timeoffset + 6 * timescale, num=len(data))
        return t[:-2], data[:-2]


