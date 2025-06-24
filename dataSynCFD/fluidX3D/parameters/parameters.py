from CoolProp.CoolProp import PropsSI
import csv


def safe_get(prop, temp, pressure, fluid):
    try:
        return PropsSI(prop, 'T', temp, 'P', pressure, fluid)
    except ValueError:
        return "N/A"
    
def get_properties(conditions):
    temp, pressure, fluid = conditions
    state = safe_get('Q', temp, pressure, fluid)
    if state == 1: # fully gas
        return "gas", "gas", "gas", "gas", "gas", "gas", state
    else :
        rho = safe_get('D', temp, pressure, fluid)  # Density [kg/m³]
        mu = safe_get('V', temp, pressure, fluid)   # Dynamic viscosity [Pa·s]
        sigma = safe_get('I', temp, pressure, fluid) # Surface tension [N/m]
        if isinstance(rho, float) and isinstance(mu, float):
            nu = mu / rho  # Kinematic viscosity [m²/s]
        else:
            nu = "N/A"
        return fluid, temp, pressure, rho, mu, sigma, nu, state

fluid_list = [
    "1-Butene", "Acetone", "Benzene", "CarbonylSulfide", "CycloHexane", "CycloPropane", "Cyclopentane", 
    "D4", "D5", "D6", "Deuterium", "Dichloroethane", "DiethylEther", "DimethylCarbonate", 
    "DimethylEther", "Ethane", "Ethanol", "EthylBenzene", "Ethylene", "EthyleneOxide", "HFE143m", "HydrogenChloride", 
    "HydrogenSulfide", "IsoButane", "IsoButene", "Isohexane", "Isopentane", "Krypton", 
    "MD2M", "MD3M", "MD4M", "MDM", "MM", "Methane", "Methanol", "MethylLinoleate", 
    "MethylLinolenate", "MethylOleate", "MethylPalmitate", "MethylStearate",
    "Neopentane", "Novec649", "OrthoDeuterium", 
    "OrthoHydrogen", "Oxygen", "ParaDeuterium", "ParaHydrogen", "Propylene", 
    "Propyne", "R11", "R113", "R114", "R115", "R116", "R12", "R123", "R1233zd(E)", 
    "R1234yf", "R1234ze(E)", "R1234ze(Z)", "R124", "R1243zf", "R125", "R13", 
    "R1336MZZE", "R134a", "R13I1", "R14", "R141b", "R142b", "R143a", "R152A", 
    "R161", "R21", "R218", "R22", "R227EA", "R23", "R236EA", "R236FA", "R245ca", 
    "R245fa", "R32", "R365MFC", "R40", "R404A", "R407C", "R41", "R410A", "R507A", 
    "RC318", "SES36", "SulfurDioxide", "SulfurHexafluoride", "Toluene", "Water", 
    "Xenon", "cis-2-Butene", "m-Xylene", "n-Butane", "n-Decane", "n-Dodecane", 
    "n-Heptane", "n-Hexane", "n-Nonane", "n-Octane", "n-Pentane", "n-Propane", 
    "n-Undecane", "o-Xylene", "p-Xylene", "trans-2-Butene"
]

# list of feasible mixing conditions; 6.85°C to 76.85°C, 1atm
conditions = [
    (temp, 1.01325e5, fluid) 
    for fluid in fluid_list
    for temp in range(280, 351, 5)
]

# write in csv
with open('properties.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['fluid','Temperature [K]', 'Pressure [Pa]', 'Density [kg/m³]', 'Dynamic Viscosity [Pa·s]', 'Surface Tension [N/m]', 'Kinematic Viscosity [m²/s]', 'State']) 
    writer.writerows(get_properties(con) for con in conditions)
