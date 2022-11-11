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
	2) Write directly to the RAM of the ad9910 chip (this example)
	
	There is probably still an unsolved issue with this code, which is described in
	https://github.com/m-labs/artiq/issues/1554
	"""
	
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

		# carrier frequency, which is the highest frequency sent to the AOM
		# and modulation depth or max. frequency difference
		self.freq_high = 80 * MHz
		self.mod_depth = 3 * MHz

		# frequency modulation limits
		self.freq_low = self.freq_high - self.mod_depth

		# create list of frequencies in Frequency Tunning Word (FTW) format
		# to write to RAM
		self.freq_index = [0.] * self.ram_length
		self.freq_ram = [0] * self.ram_length
 
		mod_depth = self.freq_high - self.freq_low
		freq_step = mod_depth / self.ram_length
 
		for i in range(self.ram_length):
			self.freq_index[i] = self.freq_low + i * freq_step
		self.dds.frequency_to_ram(self.freq_index, self.freq_ram)

		# modulation frequency
		self.freq_mod = 20 * kHz

		# sweep time
		self.sweep_time = 1 / self.freq_mod

		# time step dt, for definition see ad9910 datasheet
		self.dt = self.sweep_time / self.ram_length

		# value M from ad9910 datasheet, which is the number of timesteps in 4*ns
		self.M = int(self.dt / (4 * ns))

		# time step when running playback mode through the ram. 
		# for definition see ad9910 dds chip datasheet, but in practice 'M' is in units of 4*ns
		self.time_step = int(self.M)
 
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

		# step through ram with sample interval time
		# we only use profile 0 out of 8
		self.dds.set_profile_ram(start=0, end=self.ram_length - 1, step=self.time_step, 
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
		# the 3 osk settings are needed becaues otherwise the ram profile 0 
		# overlap with the RAM single tone, see https://github.com/m-labs/artiq/issues/1554
		self.dds.set_cfr1(internal_profile=0, 
						  ram_enable=1,
						  ram_destination=ad9910.RAM_DEST_FTW,
						  manual_osk_external=0, osk_enable=1, select_auto_osk=0) 
		self.dds.set_amplitude(1.)
		self.dds.set_att(20.*dB)
		self.dds.cpld.io_update.pulse_mu(8)
 
		# switch on DDS channel
		self.dds.sw.on()	
