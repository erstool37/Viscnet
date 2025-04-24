import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def propertycalculatorVOLUME(fraction_glyc, T):
    volume_glycerol = fraction_glyc
    volume_water = 1 - fraction_glyc
    total_volume = volume_glycerol + volume_water
    volume_fraction = volume_glycerol / total_volume

    density_glycerol = 1273.3 - 0.6121 * T
    density_water = 1000 * (1 - ((abs(T - 3.98)) / 615) ** 1.71)

    mass_glycerol = density_glycerol * volume_glycerol
    mass_water = density_water * volume_water
    total_mass = mass_glycerol + mass_water
    mass_fraction = mass_glycerol / total_mass

    viscosity_glycerol = 0.001 * 12100 * np.exp((-1233 + T) * T / (9900 + 70 * T))
    viscosity_water = 0.001 * 1.790 * np.exp((-1230 - T) * T / (36100 + 360 * T))

    a = 0.705 - 0.0017 * T
    b = (4.9 + 0.036 * T) * a ** 2.5
    alpha = 1 - mass_fraction + (a * b * mass_fraction * (1 - mass_fraction)) / (a * mass_fraction + b * (1 - mass_fraction))
    A = np.log(viscosity_water / viscosity_glycerol)
    viscosity_mix = viscosity_glycerol * np.exp(A * alpha)

    c = 1.78e-6 * T ** 2 - 1.82e-4 * T + 1.41e-2
    contraction = 1 + (c * np.sin(mass_fraction ** 1.31 * np.pi) ** 0.81)
    density_mix = (density_glycerol * fraction_glyc + density_water * (1 - fraction_glyc)) * contraction

    return density_mix, viscosity_mix

def calculate_masses_from_weight_fraction(w_g_all, V_total, T):
    rho_g = (1273.3 - 0.6121 * T) / 1000
    rho_w = (1000 * (1 - ((abs(T - 3.98)) / 615) ** 1.71)) / 1000
    density_mix = rho_g * w_g_all + rho_w * (1 - w_g_all)
    m_total = V_total * density_mix
    mass_glycerol = w_g_all * m_total
    mass_water = (1 - w_g_all) * m_total
    return mass_glycerol, mass_water

# Settings
T = 25
V_total = 80
v_g_all = np.linspace(0, 1, 10000)
eta_all = np.array([propertycalculatorVOLUME(v, T)[1] for v in v_g_all])
log_eta_all = np.log(eta_all)

# Plotting
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(v_g_all, log_eta_all, 'b')
plt.xlabel("Glycerol Volume Fraction")
plt.ylabel("log(Viscosity) [ln(Ns/m^2)]")
plt.title("log(Viscosity) vs. Glycerol Volume Fraction")
plt.grid()

plt.subplot(1, 2, 2)
plt.plot(v_g_all, eta_all, 'r')
plt.xlabel("Glycerol Volume Fraction")
plt.ylabel("Viscosity [Ns/m^2]")
plt.title("Viscosity vs. Glycerol Volume Fraction")
plt.grid()
plt.tight_layout()
plt.show()

# Interpolation
log_eta_targets = np.linspace(log_eta_all.min(), log_eta_all.max(), 13)
v_g_interp = np.interp(log_eta_targets, log_eta_all, v_g_all)

# Densities and masses
rho_interp = np.array([propertycalculatorVOLUME(v, T)[0] for v in v_g_interp])
eta_interp = np.array([propertycalculatorVOLUME(v, T)[1] for v in v_g_interp])
w_g_interp = (1273.3 - 0.6121 * T) * v_g_interp / ((1273.3 - 0.6121 * T) * v_g_interp + (1000 * (1 - ((abs(T - 3.98)) / 615) ** 1.71)) * (1 - v_g_interp))
m_glyc, m_water = calculate_masses_from_weight_fraction(w_g_interp, V_total, T)
eta_cP = eta_interp * 1000
surfT = np.full_like(w_g_interp, 0.0762)

# Save
df = pd.DataFrame({
    "WeightFraction": w_g_interp,
    "Glycerol_mass_g": m_glyc,
    "Water_mass_g": m_water,
    "Density_kg_per_m3": rho_interp,
    "Viscosity_Ns_per_m2": eta_interp,
    "Viscosity_cP": eta_cP,
    "SurfaceTension_N_per_m": surfT
})
df.to_csv("dataset/realfluid/glycerol_water_properties.csv", index=False)
print(df)