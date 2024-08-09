# calculating waist for Thorlabs scanning fabry perot interferometer
# see https://www.thorlabs.com/newgrouppage9.cfm?objectgroup_id=859

# variables
cavity_length = 50e-3  # 50 mm, distance between FP mirrors
wavelength = 689e-9  # 689 nm, wavelength
radius_curvature = 50e-3  # 50 mm
focal_length_lens = 125e-3  # lens used for focussing to FP

# waist inside the FP 
waist_fp_recommended = sqrt(wavelength/(2*π)*sqrt(cavity_length*(2*radius_curvature - cavity_length)))
println("waist fp recommended ", waist_fp_recommended/1e-3, " mm")


function incident_waist(focal_length::Float64)
    # see eq. (2) in the thorlabs page
    incident_waist = wavelength*focal_length/(π*waist_fp_recommended)
    return incident_waist
end


# calculating incident_waist
incident_waist_calculated = incident_waist(focal_length_lens)
println("incident_waist to use ", incident_waist_calculated/1e-3, " mm")
