from artiq.experiment import *


def input_led_state() -> TBool:
    return input("Enter desired LED state: ") == "1"


class SetLEDvalue(EnvExperiment):
    """Set LED Value Terminal"""
    # this runs on the host PC
    def build(self):
        self.setattr_device("core")
        self.setattr_device("led0")

    # this runs on the FPGA
    @kernel
    def run(self):
        self.core.reset()
        
        s = input_led_state()
        self.core.break_realtime()
        if s:
            self.led0.on()
        else:
            self.led0.off()
