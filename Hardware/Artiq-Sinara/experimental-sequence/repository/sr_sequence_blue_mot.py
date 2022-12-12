# author: Marijn Venderbosch
# december 2022

# public libraries
from artiq.experiment import *
from artiq.coredevice import ad9910

# user defined libraries
from sr_devices import ZylaCamera
from sr_classes import SavingExperimentalData
from sr_classes import Sr


class BlueMOT(EnvExperiment):
    """Load Blue MOT"""

    """Turn on all outputs required for running a blue MOT
    - 7 DDS static frequency
    - 1 DDS frequency comb spectrum (red MOT)"""

    def build(self):  
        self.setattr_device("core")

        # global attenuation, for all DDS chips the same
        self.setattr_argument("Attenuation", NumberValue(10*dB, unit='dB', scale=dB, min=0*dB, max=31*dB))

        """initialize 4 + 3 DDS channels"""
        # initialize and set channel, frequency, amplitude for channels 0, 1, 2 and 3 on DDS0 (urukul0)
        # set number of channels used 
        self.nr_channels_dds0 = 4

        for ch in range(self.nr_channels_dds0):

            # initializes all channels (0,1,2,3) on urukul0 (first one)
            self.setattr_device(f"urukul0_ch{ch}")

            # This two attributes will be shown in the GUI grouped by channel
            # use/don't use each channel
            self.setattr_argument(f"State_DDS0_Channel{ch}", 
                                  BooleanValue(ch == 0),
                                  f"DDS0_Channel{ch}")

            # each channel's frequency, between min. and max value
            self.setattr_argument(f"Frequency_DDS0_Channel{ch}",
                                  NumberValue(100.0*MHz, unit='MHz', scale=MHz, min=70*MHz, max=220*MHz),
                                  f"DDS0_Channel{ch}")

            # each channel's amplitude, between 0 and 1 (exponential scale)
            self.setattr_argument(f"Amplitude_DDS0_Channel{ch}",
                                  NumberValue(0.1, min=0., max=1.),
                                  f"DDS0_Channel{ch}")

        # initialize and set channel, frequency, amplitude for channels 0, 1 and 2 on DDS1 (urukul1)
        # set number of channels used 
        self.nr_channels_dds1 = 3

        for ch in range(self.nr_channels_dds1):

            # initializes all channels (0,1,2) on urukul1 (second one)
            # channel 3 of urukul1 is reserved for the red MOT frequency comb
            # for comments see for loop over DDS0
            self.setattr_device(f"urukul1_ch{ch}")

            self.setattr_argument(f"State_DDS1_Channel{ch}",
                                  BooleanValue(ch == 0),
                                  f"DDS1_Channel_{ch}")

            self.setattr_argument(f"Frequency_DDS1_Channel{ch}",
                                  NumberValue(100.0*MHz, unit='MHz', scale=MHz, min=70*MHz, max=220*MHz),
                                  f"DDS1_Channel_{ch}")

            self.setattr_argument(f"Amplitude_DDS1_Channel{ch}",
                                  NumberValue(0.1, min=0., max=1.),
                                  f"DDS1_Channel_{ch}")

        # group DDS on/off state in single variable
        self.all_states = [self.State_DDS0_Channel0,
                           self.State_DDS0_Channel1,
                           self.State_DDS0_Channel2,
                           self.State_DDS0_Channel3,
                           self.State_DDS1_Channel0,
                           self.State_DDS1_Channel1,
                           self.State_DDS1_Channel2
                           ]

        # do the same for amplitude (amp), frequencies (freq) and channel variables
        self.all_amps = [self.Amplitude_DDS0_Channel0,
                         self.Amplitude_DDS0_Channel1,
                         self.Amplitude_DDS0_Channel2, 
                         self.Amplitude_DDS0_Channel3,
                         self.Amplitude_DDS1_Channel0, 
                         self.Amplitude_DDS1_Channel1,
                         self.Amplitude_DDS1_Channel2
                         ]

        self.all_freqs = [self.Frequency_DDS0_Channel0,
                          self.Frequency_DDS0_Channel1, 
                          self.Frequency_DDS0_Channel2,
                          self.Frequency_DDS0_Channel3,
                          self.Frequency_DDS1_Channel0, 
                          self.Frequency_DDS1_Channel1, 
                          self.Frequency_DDS1_Channel2
                          ]

        self.all_channels = [self.urukul0_ch0, 
                             self.urukul0_ch1,
                             self.urukul0_ch2, 
                             self.urukul0_ch3,
                             self.urukul1_ch0, 
                             self.urukul1_ch1, 
                             self.urukul1_ch2
                             ]
   
        # use grouped variables
        self.use_amps = []
        self.use_freqs = []
        self.use_channels = []

        # zip in single iterable
        for state, ch_n, freq_n, amp_n in zip(self.all_states, self.all_channels, self.all_freqs, self.all_amps):
            
            # enable frequency and amplitude if state = 1
            if state:
                self.use_channels.append(ch_n)
                self.use_freqs.append(freq_n)
                self.use_amps.append(amp_n)

        """initialize red MOT DDS"""
        # 2nd Urukul module
        self.setattr_device("urukul1_cpld") 

		# 2nd Urukul module, channel 0 (first one)
        self.setattr_device("urukul1_ch3")
        self.red_mot_dds = self.urukul1_ch3

		# load variables
        self.setattr_argument("on_state", 
                              BooleanValue(True), 
                              "red_MOT_DDS")

        self.setattr_argument("center_frequency",
                              NumberValue(100.0*MHz, unit='MHz', scale=MHz, min=70*MHz, max=220*MHz),
                              "red_MOT_DDS")
		
        self.setattr_argument("modulation_depth",
                              NumberValue(3.0*MHz, unit='MHz', scale=MHz, min=0.1*MHz, max=10*MHz),
                              "red_MOT_DDS")

        self.setattr_argument("modulation_frequency",
							  NumberValue(20*kHz, unit='kHz', scale=kHz, min=1*kHz, max=500*kHz), 
                              "red_MOT_DDS")

        self.setattr_argument("rf_amplitude",
							  NumberValue(0.2, min=0., max=1.0), 
                              "red_MOT_DDS")

    def prepare(self):

		# ram size in bits
        self.ram_length = 1024

		# frequency modulation limits
        self.freq_low = self.center_frequency - self.modulation_depth

		# create list of frequencies in Frequency Tunning Word format to write to RAM
        self.freq_index = [0.] * self.ram_length
        self.freq_ram = [0] * self.ram_length
        freq_step = self.modulation_depth / self.ram_length
 
        for i in range(self.ram_length):
            self.freq_index[i] = self.freq_low + i * freq_step
        self.red_mot_dds.frequency_to_ram(self.freq_index, self.freq_ram)

		# sweep time
        self.sweep_time = 1 / self.modulation_frequency

		# time step dt, for definition see ad9910 datasheet
        self.dt = self.sweep_time / self.ram_length

		# value M from ad9910 datasheet, which is the number of timesteps in 4*ns
        self.M = int(self.dt / (4 * ns))

		# time step when running playback mode through the ram. 
        self.time_step = int(self.M)

    @kernel  
    def run(self):
        self.core.reset()

        """turn on 7 DDS channels"""
        # For some reason this delay is needed, otherwise does not work. Should be ~29 ms at least?
        delay(100 * ms)

        # initialises CPLD all the selected channels
        for channel in self.use_channels:
            channel.cpld.init()
            channel.init()

        delay(10*ms)

        for i in range(len(self.use_channels)):

            # Writes global attenuation and specific amplitude, frequency variables to each Urukul
            self.use_channels[i].set_att(self.Attenuation)

            # self.use_channels[i].set_amplitude(self.use_amps[i])
            self.use_channels[i].set(self.use_freqs[i], amplitude=self.use_amps[i])

        # turn on every selected channel
        # only turned on when state=1
        for nr in range(self.nr_channels_dds0 + self.nr_channels_dds1):
            if self.all_states[nr] == 1:
                self.all_channels[nr].sw.on()
            else:
                self.all_channels[nr].sw.off()
    
        """enable red MOT DDS"""
        # initialize DDS
        # delay before/after are needed. Not sure why this is. 
        delay(50*ms)
        self.red_mot_dds.cpld.init()
        delay(50*ms)
        self.red_mot_dds.init()

        # enact with I/O update pulse
        self.red_mot_dds.cpld.io_update.pulse(100 * ns)
        self.core.break_realtime()

        # prepare ram profile, disable RAM for writing data, enact with I/O pulse
        self.red_mot_dds.cpld.io_update.pulse_mu(8) 

        # step through ram with sample interval time, only using profile 0 of 8
        # enact with I/O pulse
        self.red_mot_dds.set_profile_ram(start=0, end=self.ram_length - 1, step=self.time_step, 
                                profile=0, 
                                mode=ad9910.RAM_MODE_CONT_RAMPUP)
        self.red_mot_dds.cpld.set_profile(0)
        self.red_mot_dds.cpld.io_update.pulse_mu(8)

        # write data to RAM
        delay(100 * us)
        self.red_mot_dds.write_ram(self.freq_ram)
        delay(100 * us)

        # only enable DDS output when 'on_state' is checked
        if self.on_state == True:

            # enable RAM mode
            self.red_mot_dds.set_cfr1(internal_profile=0, 
                            ram_enable=1,
                            ram_destination=ad9910.RAM_DEST_FTW,
                            manual_osk_external=0, osk_enable=1, select_auto_osk=0) 
            
            # set amplitude, attenuation and enact with I/O pulse
            self.red_mot_dds.set_amplitude(self.rf_amplitude)
            self.red_mot_dds.set_att(self.Attenuation)
            self.red_mot_dds.cpld.io_update.pulse_mu(8)
            
            # turn on DDS channel
            self.red_mot_dds.sw.on()	
