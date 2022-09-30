from artiq.experiment import *


class TTLoutputOnOffPulse(EnvExperiment):
    """TTL Output On Off"""

    """
    Code outputs single TTL pulse on output 4 on the MCX board
    First sets output to high, then low. Finally, pulses it, which is equivalent. 
    """
    # this runs on the host
    def build(self):  # this runs on the host

        self.setattr_device("core")
        self.setattr_device("ttl4")

    @kernel  # this runs on the FBGA
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
