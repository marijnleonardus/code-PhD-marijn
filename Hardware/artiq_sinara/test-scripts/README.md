# Test script collection

## Description

This folder contains simple test scripts that the user can try and edit. 

## Contents

* `LED.py`: the simplest possible script that you can run on Artiq. If something does not work anymore, run this file.

* `Repository`: the contents of this folder show up in the Artiq Dashboard explorer. It contains:
    * LED scripts. These show how to work with parallel/sequential sequences in Artiq. As well as how to run a script that takes input from the user first. 
        * `led_parallel_sequential.py`
        * `set_LED_state.py`
    * DMA (direct memory access) example, for running extremely fast sequential pulses. 
        * `dma_example.py` 
    * DDS scripts, for single channel, multi channel outputs, as well as frequency modulation.
        * `dds_single_channel.py`
        * `dds_multi_channel.py`
        * `dds_ramp_test_ram.py`
    * Fastino script, for ramping a voltage.
        * `fastino_dac_voltage_ramp.py`


