# author: Marijn Venderbosch
# october 2022


from artiq.experiment import *
from artiq.coredevice import ad9910
import numpy as np

 
class DDSFrequencyRampRAM(EnvExperiment):
	"""DDS Frequency Ramp RAM

	example adapted from https://github.com/m-labs/artiq/issues/1378

	There are two methods of doing a DDS frequency ramp:
	1) Direct ramp generator (DRG), see https://forum.m-labs.hk/d/120-sweep-frequency-dds/2
	2) Write directly to the RAM of the ad9910 chip (this example)"""
	
	def build(self):
		self.setattr_device("core")

		# 2nd Urukul module
		self.setattr_device("urukul1_cpld") 

		# 2nd Urukul module, channel 0 (first one)
		self.setattr_device("urukul1_ch0")
		self.dds = self.urukul1_ch0
 
	def prepare(self):

		# ram size in bits
		self.ram_length = 1024

		# start and end frequency to ramp over
		self.freq_low = 79.5 * MHz
		self.freq_high = 80.5 * MHz

		# sample interval associated with playback rate:
		# sample_interval = 1 / playback_rate
		self.sample_interval = int(1e8)

		# create list of frequencies in Frequency Tunning Word (FTW) format
		# to write to RAM
		self.freq_index = [0.] * self.ram_length
		self.freq_ram = [0] * self.ram_length
 
		freq_span = self.freq_high - self.freq_low
		freq_step = freq_span / self.ram_length
 
		for i in range(self.ram_length):
			self.freq_index[i] = self.freq_low + i * freq_step
		self.dds.frequency_to_ram(self.freq_index, self.freq_ram)
 
	@kernel
	def run(self):
		self.core.reset()
		
		# initialize DDS
		self.dds.cpld.init()
		self.dds.init()
		self.dds.cpld.io_update.pulse(100 * ns)
		self.core.break_realtime()

		# prepare ram profile
		# disable RAM for writing data
		self.dds.set_cfr1() 

		# I/O pulse to enact RAM change
		self.dds.cpld.io_update.pulse_mu(8) 

		# step through ram with samplint interval time
		# we only use profile 0 out of 8
		self.dds.set_profile_ram(start=0, end=self.ram_length - 1, step=self.sample_interval, 
								 profile=0, 
								 mode=ad9910.RAM_MODE_CONT_RAMPUP)
		self.dds.cpld.set_profile(0)

		# I/O pulse to enact RAM change
		self.dds.cpld.io_update.pulse_mu(8)
 
		# write data to RAM
		delay(100 * us)
		self.dds.write_ram(self.freq_ram)
		delay(100 * us)
 
		# enable RAM mode (enacted by IO pulse) and fix other parameters
		self.dds.set_cfr1(internal_profile=0, 
						  ram_destination=ad9910.RAM_DEST_FTW,
						  ram_enable=1)
		self.dds.set_amplitude(1.)
		self.dds.set_att(30.*dB)
		self.dds.cpld.io_update.pulse_mu(8)
 
		# switch on DDS channel
		self.dds.sw.on()	
		