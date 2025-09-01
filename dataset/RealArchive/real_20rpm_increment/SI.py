import json, glob, os

src_dir = "parameters"   # change this

# find all JSON files in the folder
json_files = glob.glob(os.path.join(src_dir, "*.json"))

for fp in json_files:
    with open(fp, "r") as f:
        d = json.load(f)

    # Convert dynamic viscosity from cP → Pa·s
    if "dynamic_viscosity" in d:
        d["dynamic_viscosity"] = float(d["dynamic_viscosity"]) * 1e-3

    # Convert kinematic viscosity from cSt → m²/s
    if "kinematic_viscosity" in d:
        d["kinematic_viscosity"] = float(d["kinematic_viscosity"]) * 1e-3

    # density (kg/m³) and surface_tension (N/m) already SI → unchanged

    with open(fp, "w") as f:
        json.dump(d, f, indent=2)

    print(f"Converted: {fp}")