import json
from pathlib import Path

base_dir = Path("parameters")

for json_file in base_dir.glob("*.json"):
    try:
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        if "RENDER" in data and "kinematic_viscosity" in data:
            original_val = data["kinematic_viscosity"]
            data["kinematic_viscosity"] = original_val / 10

            with open(json_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)

            print(f"Updated {json_file.name}: {original_val:.6e} → {data['kinematic_viscosity']:.6e}")

    except Exception as e:
        print(f"Error processing {json_file.name}: {e}")