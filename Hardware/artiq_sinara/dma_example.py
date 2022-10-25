from artiq.experiment import *


class DirectMemory(EnvExperiment):
    """Direct memory access TTL
    
    example for making TTL pulses spaced 100 ns in time using the
    direct memory access (DMA)"""

    """this code runs on the host PC"""
    def build(self):

        self.setattr_device("core")
        self.setattr_device("core_dma")
        self.setattr_device("ttl4")

    """this code runs on the FPGA"""
    @kernel
    def record(self):
        with self.core_dma.record("pulses"):

            # all RTIO operations now go to the "pulses"
            # DMA buffer, instead of being executed immediately.
            for i in range(50):
                self.ttl4.pulse(100*ns)
                delay(100*ns)

    @kernel
    def run(self):
        self.core.reset()
        self.record()

        # prefetch the address of the DMA buffer
        # for faster playback trigger
        pulses_handle = self.core_dma.get_handle("pulses")
        self.core.break_realtime()

        while True:
            # execute RTIO operations in the DMA buffer
            # each playback advances the timeline by 50*(100+100) ns
            self.core_dma.playback_handle(pulses_handle)
            self.core.break_realtime()