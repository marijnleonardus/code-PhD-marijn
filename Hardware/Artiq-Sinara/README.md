# Artiq - Sr machine

Code for running the Sr Setup using the Artiq Kasli EEM controller.
There are two repositories:

* `test-scripts`: contains simple working examples of different things we can do with artiq; e.g. a frequency ramp, voltage ramp, single and multiple outputs on one DDS chip.
* `experimental-sequence`: actual real-world experimental sequence that can be run on the Sr machine. 

Both folders share the same `devibe_db.py` file.

## Done

- [x] get trivial LED script running
- [x] get TTL pulses working
- [x] get DDS working and calibrate RF power output
- [x] make arbitrary voltage ramps using Fastino DAC
- [x] TTL override working
- [x] frequency modulation using ad9910 DDS chip
- [x] replace shutter driver by ArtiQ TTL pulse

## Roadmap

Practically, the following things still have to be done as well:

- [ ] replace Moglabs driver by ArtiQ DDS
- [ ] write a more elaborate script, e.g. blue MOT script
- [ ] create GUI for visualizing experimental sequenes

Final goal: stand alone program for controlling all time dependent hardware on Sr machine

## Sinara boards

* *Kasli* EEM Controller (FPGA)
* two 2238 cards (16 channel MCX TTL)
* two *Urukul* 4 channel DDS cards
* Camera grabber
* 8 channnel EEM (sampler)
* DAC (*Fastino*)
* four 8 channel BNC-IDC adapters

## Running ArtiQ

Can be run from both Linux and Windows PC. The first instructions that follow are therefore OS specific.

### Linux

#### 1. Change directory

Replace 'USERNAME' by actual username and 'FOLDER' by either `test-scripts` or `experimental-sequence`. The directory can be any folder but should contain `device_db.py` file, e.g.


```
$ cd /home/USERNAME/Documents/Sr_Artiq/FOLDER/
```

#### 2. Activate environment that contains correct packages

```
$ nix shell
```

### Windows

#### 1. Change directory
Open a windows cmd window and type the following

```
cd Documents\Github\artiq-sr-machine\FOLDER
```

#### 2. Load environment that contains the correct pacakges


```
conda activate artiq
```

#### 3. Running scripts

You can now run scripts by typing in the terminal:
```
artiq_run script.py
```

If you want to use the dashboard (GUI), run the folloing two commands.

#### 4. Load artiq master


```
$ artiq_master
```

#### 5. Dashboard.

Opens up dashboard. Open a new command window as the previous command blocks the user from typing new entries.

```
$ artiq_dashboard
```

To combine steps 4. and 5. without opening another windows, instead of `artiq_master, artiq_dashboard`, type just 

```
artiq_session
```

The experiments that can be run should be saved under the `~/repository` folder.
Upon changing the experiment file, reload by right clicking on the explorer in the dashboard and select `scan repository HEAD`


## Contributing to the code

See `CONTRIBUTING.md`.

## Learning more
Artiq-Language: https://m-labs.hk/artiq/manual/introduction.html 

Hardware-wiki: https://github.com/sinara-hw/meta/wiki

Gitlab: https://docs.gitlab.com/ee/topics/git/terminology.html
