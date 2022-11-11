from artiq.experiment import *


class LED(EnvExperiment):
    """LED Parallel Sequential"""

    """shows difference between running parallel and sequential"""
    def build(self):
        self.setattr_device("core")
        self.setattr_device("led0")
        self.setattr_device("led1")

    @kernel
    def run(self):
        self.core.reset()
        with parallel:
            with sequential:
                self.led0.on()
                delay(2*s)
                self.led0.off()
            with sequential:
                self.led1.on()
                delay(1*s)
                self.led1.off()
