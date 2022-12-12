# author: Marijn Venderbosch
# october 2022

from artiq.experiment import *


class FastinoDACRamp(EnvExperiment):
    """Fastino DAC Ramp

    Script for ramping a voltage of the DAC.
    Any arbitrary of ramps, timings and begin/end voltages 
    can be constructed using this example
    """

    """this code runs on the lab PC"""
    def build(self): 
        # initalize core as well as fastino0 DAC

        self.setattr_device("core")
        self.setattr_device("fastino0")  

    """this code is prepared in parallel with the code ran on the FPGA"""
    def prepare(self): 

        # define time to ramp over as well as number of steps
        self.ramp_time = 250 * ms
        self.number_steps = 100
        
        # the time of a single step is now defined as
        self.time_step = self.ramp_time / self.number_steps

        # define min. and max. voltage for ramp (start and end voltage)
        self.max_voltage = 5
        self.min_voltage = 0

        #  defines voltage ramp in machine units, steps from -10V to 10V
        self.voltages_up = [(self.max_voltage - self.min_voltage) / self.number_steps * i 
            for i in range(self.number_steps)]

        self.voltages_down = [self.max_voltage- ((self.max_voltage - self.min_voltage) / self.number_steps * j)
            for j in range(self.number_steps)]

        # set Fastino DAC output channel number
        self.channel_number = 0

        # counter that will be used to keep track of time later
        self.counter_up = 0
        self.counter_down=0

    """this code runs on the FPGA"""
    @kernel  
    def run(self):
        self.core.reset()
        self.core.break_realtime()
        self.fastino0.init()

        # 200us delay, to prevent underflow
        delay(200 * us)

        # loops until manually broken(from bash terminal, this requires closing terminal)
        while self.counter_up < self.number_steps:

            # loops over all voltages in voltages_mu list

            for voltage in self.voltages_up:
                #  write voltages to output channel of fastino DAC
                self.fastino0.set_dac(self.channel_number, voltage)

                delay(self.time_step)

                # increase counter for next iteration loop
                self.counter_up = self.counter_up + 1
        
        delay(500 * ms)

        while self.counter_down < self.number_steps:

            # again loops, this time over different voltage list
            for voltage in self.voltages_down:

                #  write voltages to output channel of fastino DAC
                self.fastino0.set_dac(self.channel_number, voltage)

                delay(self.time_step)

                # increase counter
                self.counter_down = self.counter_down + 1

        


