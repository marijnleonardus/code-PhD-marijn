# estimate number of photons incident on EMCCD camera and number of photons scattered from atoms
# from the number of counts recorded by the Ixon 888 Ultra EMCCD camera

# %% variables
# for camera counts, see main function

emgain = 300 # electron multiplier gain
quantum_efficiency = 0.8 # @ 461 nm
sensitivity = 15.8 # electrons per A/D count for pre-amplifier mode '1' (see datasheet)
offset = 500 # for dark image, gaussian distribution peaks here 

## optical path efficincies
collection_efficiency = 0.067 # isotropic emission, NA0.5, 15 mm working distance
objective_transmission = 0.85 # @ 461 nm 
glass_cell_transmission = 0.95  # uncoated
optical_element_transmission = 0.995 # A coated optical element @ 461 nm 

# %% functions


def calculate_photons_counted(cam_counts):
    photoelectrons = (cam_counts - offset)*sensitivity/emgain
    return photoelectrons


def calculate_scattered_photons(photons_counted):
    optical_path_transmission = glass_cell_transmission**2*objective_transmission*optical_element_transmission**8
    scattered_photons = photons_counted/collection_efficiency/optical_path_transmission/quantum_efficiency
    return scattered_photons


# %% execute script


if __name__ == "__main__":
    cam_counts = 10e3
    photons_counted = calculate_photons_counted(cam_counts)
    print("nr photons in measurment:", photons_counted)

    scattered_photons = calculate_scattered_photons(photons_counted)
    print("nr photons scattered", scattered_photons)
