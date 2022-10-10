from pylcp_class import RecoilLimitedMOT

def main():
    #  independent variables
    wavenumber = 2 * np.pi * 689e-7  # 1/cm
    linewidth = 7.4e3  # 1/s, thus kHz linewidth
    detuning = -200/7.5
    saturation = 25
    bohr_magneton = cts.value('Bohr magneton in Hz/T')
    atomic_mass_unit = cts.value('atomic mass constant')

    #  dependent variables
    length = 1/wavenumber  # cm
    time = 1/linewidth  # s
    alpha = 1.5 * bohr_magneton * 1e-4 * 8 * length / (linewidth / (2 * pi))
    mass = 87.8 * atomic_mass_unit * (length * 1e-2)**2 / hbar / time
    gravity_vector = -np.array([0., 0., 9.8 * time**2 / (length * 1e-2)])

    print(mass, time, length, alpha, gravity_vector)

    #  create object
    sr_red_mot = RecoilLimitedMOT(detuning, saturation, length, time, alpha)


if __name__ == '__main__':
    main()
