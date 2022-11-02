from artiq.experiment import *


class SetAllUrukulFrequencies(EnvExperiment):
    """Urukul frecuencies setter"""

    """Set the frecuencies/amplitudes of every Urukul channel
    Script based on example from the internet"""
    def build(self):  # runs on host PC
        self.setattr_device("core")

        # global attenuation
        self.setattr_argument("attenuation",
                              NumberValue(0*dB, unit='dB', scale=dB, min=0*dB, max=31*dB),
                              )

        # for loop over all channels
        for ch in range(4):
            # initializes all channels (0,1,2,3) on urukul0 (first one)
            self.setattr_device(f"urukul0_ch{ch}")

            # This two attributes will be shown in the GUI grouped by channel
            # use/don't use each channel
            self.setattr_argument(f"state_ch{ch}",
                                  BooleanValue(ch == 0),
                                  f"canal_{ch}")

            # each channel's frequency
            self.setattr_argument(f"freq_ch{ch}",
                                  NumberValue(100.0*MHz, unit='MHz', scale=MHz, min=70*MHz, max=220*MHz),
                                  f"canal_{ch}")

            # each channel's amplitude
            self.setattr_argument(f"amp_ch{ch}",
                                  NumberValue(0.1, min=0., max=1.),
                                  f"canal_{ch}")

        self.all_amps = [self.amp_ch0, self.amp_ch1, self.amp_ch2, self.amp_ch3]
        self.all_freqs = [self.freq_ch0, self.freq_ch1, self.freq_ch2, self.freq_ch3]
        self.states = [self.state_ch0, self.state_ch1, self.state_ch2, self.state_ch3]
        self.all_channels = [self.urukul0_ch0, self.urukul0_ch1, self.urukul0_ch2, self.urukul0_ch3]

        self.use_amps = []
        self.use_freqs = []
        self.use_channels = []

        for state, ch_n, freq_n, amp_n in zip(self.states, self.all_channels,
                                              self.all_freqs, self.all_amps):
            if state:
                self.use_channels.append(ch_n)
                self.use_freqs.append(freq_n)
                self.use_amps.append(amp_n)

    @kernel  # runs on FPGA
    def run(self):
        self.core.reset()

        # For some reason this delay is needed, otherwise does not work. Should be ~29 ms at least?
        delay(100 * ms)

        # initialises CPLD all the selected channels
        for channel in self.use_channels:
            channel.cpld.init()
            channel.init()

        delay(10*ms)

        for i in range(len(self.use_channels)):

            # Writes globak attenuation and specific amplitude, frequency variables to each Urukul
            self.use_channels[i].set_att(self.attenuation)

            # self.use_channels[i].set_amplitude(self.use_amps[i])
            self.use_channels[i].set(self.use_freqs[i], amplitude=self.use_amps[i])

        # turn on every selected channel
        for nr in range(4):
            if self.states[nr] == 1:
                self.all_channels[nr].sw.on()
            else:
                self.all_channels[nr].sw.off()
