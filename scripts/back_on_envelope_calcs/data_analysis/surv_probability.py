image_fidelity = 0.997
surv_probability = 0.99
filling_fraction = 0.46


def corrected_surv_prob(image_fid, surv_prob, filling_frac):
    F = image_fid
    S0 = surv_prob
    p = filling_frac

    # 
    S = (S0 + F - 1)*(p*F + (1 - p)*(1 - F))/(p*F*(2*F - 1))
    return S


print(corrected_surv_prob(image_fidelity, surv_probability, filling_fraction))
