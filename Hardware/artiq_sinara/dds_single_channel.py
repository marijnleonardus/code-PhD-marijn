from artiq.experiment import*


class UrukulFrequencySelectable(EnvExperiment):
    """Urukul Selectable Frequency

    Sets user-defined frequency and amplitude to the Urukul DDS device
    Variables are set in the GUI (Artiq dashboard)"""

    """this runs on the host device"""
    def build(self):  
        
        # initialize core and urukul0 (first one), channel 0
        self.setattr_device("core")                                                    
        self.setattr_device("urukul1_ch0")    

        # dashboard input in MHz                                         
        self.setattr_argument("freq", 
                              NumberValue(ndecimals=0, unit="MHz", step=1)
                              )

        # attenuation; between 0 and 31 dB
        self.setattr_argument("attenuation",
                              NumberValue(10*dB, unit='dB', scale=dB, min=0*dB, max=31*dB),
                              )

        # amplitude; between 0 and 1
        self.setattr_argument("amplitude",
                              NumberValue(0.1, ndecimals=1, min=0., max=1.),
                              )
    
    """this runs on the FPGA"""
    @kernel  
    def run(self):  
        self.core.reset() 

        delay(50*ms)                                            

        # initialises CPLD on channel 0
        self.urukul1_ch0.cpld.init()                            
        self.urukul1_ch0.init()                                

        delay(10*ms)                                                                          

        # write attenuatin to urukul channel
        self.urukul1_ch0.set_att(self.attenuation)        

        # switches urukul channel on           
        self.urukul1_ch0.sw.on()                                
        self.urukul1_ch0.set(self.freq, amplitude=self.amplitude)   

        delay(20*s)          

        # switch off
        self.urukul1_ch0.sw.off()                               
