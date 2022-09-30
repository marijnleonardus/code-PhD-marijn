from artiq.experiment import*


class UrukulFrequencyPulse(EnvExperiment):
    """Urukul Single Frequency Pulse"""

    """Code outputs fixed frequency and amplitude set in the script on the Urukul DDS device"""
    """freq and amp variables are set in the script itself"""
    def build(self):  # This code runs on the host device

        self.setattr_device("core")                             # sets core device drivers as attributes
        self.setattr_device("urukul0_ch0")                      # sets urukul0, channel 1 device drivers as attributes

    @kernel  #  This code runs on the FPGA
    def run(self):  
        self.core.reset()                                       #  resets core device

        delay(29*ms)

        self.urukul0_ch0.cpld.init()                            #  initialises CPLD on channel 1
        self.urukul0_ch0.init()                                 #  initialises channel 1
        delay(10 * ms)                                          #  10ms delay
        
        freq = 100*MHz                                          #  defines frequency variable
        amp = 1.0                                               #  defines amplitude variable as (scaled 0-1)
        attenuation= 1.0                                        #  defines attenuation variable
        
        self.urukul0_ch0.set_att(attenuation)                   #  writes attenuation to urukul channel
        self.urukul0_ch0.sw.on()                                #  switches urukul channel on

        self.urukul0_ch0.set(freq, amplitude=amp)               #  writes frequency and amplitude variables to urukul
        delay(2*s)                                              #  2s delay
        self.urukul0_ch0.sw.off()                               #  switches urukul channel off
