import numpy as np
from PIL import Image
import os

magnification = 0.6 # obtain from device db
pixel_size = 3.45e-6 # ,,
wavelength = 461e-9 # ,,
bin_size = 2 # ,,

# compute atom nr from size, amplitude, cross sec
cross_section = 3*wavelength**2/(2*np.pi)
px_area = (pixel_size*bin_size/magnification)**2


folder = r"Z:\Strontium\Images\2025-11-21"
RID = ["scan132558", "scan132826", "scan133018", "scan133215", "scan133450", 'scan133641', "scan133911", "scan134107", "scan134259", "scan134454", "scan134655", "scan134857", "scan135055", "scan135256", "scan135451", "scan135646", "scan135836", "scan140032", "scan140227", "scan140436"]
power = (256 - np.arange(256, 56, -10))*0.3077

def compute_atom_number(RID):
    rid_path = os.path.join(folder, RID)

    # Find all relevant images
    image_files = [
        f for f in os.listdir(rid_path)
        if f.endswith("absorption.tif")
    ]

    atom_numbers = []

    for file in image_files:
        full_path = os.path.join(rid_path, file)

        with Image.open(full_path) as img:
            od_image = (-200 + np.array(img, dtype=np.float64)) / 1000
            nr_atoms = px_area*(1/cross_section)*np.sum(od_image)
            atom_numbers.append(nr_atoms)

    atom_numbers = np.array(atom_numbers)

    mean_atoms = np.mean(atom_numbers)
    sem_atoms = np.std(atom_numbers, ddof=1) / np.sqrt(len(atom_numbers))

    return mean_atoms, sem_atoms


bg, sbg = compute_atom_number("scan132558")

N_atoms = []
S_atoms = []

for r in RID:
    
    N_atoms.append(compute_atom_number(r)[0])
    S_atoms.append(compute_atom_number(r)[1])
  
# save result
number_atoms_array = N_atoms - bg
errors_number_atoms = S_atoms
np.savetxt('output/processed_data/number_atoms_array.npy', number_atoms_array)
np.savetxt('output/processed_data/errors_number_atoms.npy', errors_number_atoms)
np.savetxt('output/processed_data/power_array.npy', power)
