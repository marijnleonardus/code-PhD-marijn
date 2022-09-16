# authors: Marijn L. Venderbosch
# september 2022

"""
One of simplest examples 

code demonstrates how to use a TTL pulse on the ArtiQ.
Uses as output TTL4 on the MCX board

First sets output to high and then low
Next pulses it
"""

from artiq.experiment import *

class TTL_Output_On_Off_Pulse(EnvExperiment):
    # this runs on the host
    def build(self):

        self.setattr_device("core")
        self.setattr_device("ttl4")

    @kernel # this runs on the FBGA
    def run(self):

        self.core.reset()                      
        self.ttl4.output()
        delay(1*us)

        self.ttl4.on()
        delay(500*ms)
        self.ttl4.off()
        delay(500*ms)

        self.ttl4.pulse(500*ms)
        delay(200*ms)