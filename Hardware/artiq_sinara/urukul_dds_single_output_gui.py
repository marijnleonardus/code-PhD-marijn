from artiq.experiment import*


class UrukulFrequencySelectable(EnvExperiment):
    """Urukul Selectable Frequency"""

    """Sets user-defined frequency and amplitude to the Urukul DDS device"""
    """Variables are set in the GUI (Artiq dashboard)"""
    def build(self):  #  This code runs on the host device
        
        self.setattr_device("core")                                                    #  sets core device drivers
        self.setattr_device("urukul0_ch0")                                             #  sets urukul0, channel 1 device
        self.setattr_argument("freq", NumberValue(ndecimals=0, unit="MHz", step=1))    #  dashboard input in MHz
    
    @kernel  #  This code runs on the FPGA
    def run(self):  
        self.core.reset() 
        delay(50*ms)                                            #  resets core device
        self.urukul0_ch0.cpld.init()                            #  initialises CPLD on channel 1
        self.urukul0_ch0.init()                                 #  initialises channel 1
        delay(10*ms)                                            #  10ms delay
        
        amp = 1.0                                               #  defines amplitude variable (scaled 0-1)
        attenuation = 1.0                                       #  defines attenuation variable
        
        self.urukul0_ch0.set_att(attenuation)                   #  writes attenuation to urukul channel
        self.urukul0_ch0.sw.on()                                #  switches urukul channel on

        self.urukul0_ch0.set(self.freq, amplitude=amp)          #  writes frequency and amplitude variables to urukul
        delay(2*s)                                              #  2s delay
        self.urukul0_ch0.sw.off()                               #  switches urukul channel off
