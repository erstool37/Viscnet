import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Constants
T = 25
V_total = 80  # mL

def propertycalculatorVOLUME(fraction_glyc, T):
    vg = fraction_glyc
    vw = 1 - vg

    rho_g = 1273.3 - 0.6121 * T
    rho_w = 1000 * (1 - (abs(T - 3.98) / 615) ** 1.71)

    m_g = rho_g * vg
    m_w = rho_w * vw
    m_total = m_g + m_w
    mf_g = m_g / m_total

    eta_g = 0.001 * 12100 * np.exp((-1233 + T) * T / (9900 + 70 * T))
    eta_w = 0.001 * 1.790 * np.exp((-1230 - T) * T / (36100 + 360 * T))

    a = 0.705 - 0.0017 * T
    b = (4.9 + 0.036 * T) * a ** 2.5
    alpha = 1 - mf_g + (a * b * mf_g * (1 - mf_g)) / (a * mf_g + b * (1 - mf_g))
    A = np.log(eta_w / eta_g)
    eta_mix = eta_g * np.exp(A * alpha)

    c = 1.78e-6 * T**2 - 1.