"""
Feasibility scan for the 5-tube stage used to generate the stage surrogate
training data. For each (cation, tube 1 feed concentration, tube 1 pressure)
the stage is built and solved from scratch; failures are recorded instead of
stopping the scan. Results are appended to stage_feasibility_scan.csv as they
are produced, so a partial scan is still useful.

Run from this directory:
    python scan_stage_feasibility.py                      # full default scan
    python scan_stage_feasibility.py La 75 11,12,13       # one cation, concentrations
                                                          # (comma separated), pressures

Author: Claude
"""

import csv
import os
import sys
import traceback

from generate_synthetic_data import build_stage_model, get_stage_tube_conditions

OUTFILE = "stage_feasibility_scan.csv"

# tube 1 applied pressures (bar) and feed concentrations (mol/m3) to scan.
# Low concentrations are the most likely to dry the stage at high pressure,
# high concentrations are the most likely to fail at low pressure.
SCAN = {
    "Na": {"pressures": [3, 4, 5, 6, 7, 8], "concentrations": [150, 200]},
    "Ca": {"pressures": [6, 7, 8, 9, 10, 11, 12], "concentrations": [150, 200]},
    "La": {"pressures": [10, 11, 12, 13, 14, 15, 16], "concentrations": [50, 100]},
}

FIELDS = [
    "cation",
    "tube_1_pressure",
    "tube_1_feed_conc",
    "status",
    "min_tube_retentate_flow",
    "min_tube_inlet_flow",
    "max_tube_inlet_conc",
    "tube_retentate_flows",
    "message",
]


def scan_case(cation, concentration, pressure, num_tubes=5):
    # same stage settings as generate_single_salt_stage_data
    chloride_multiplier = {"Na": 1, "Ca": 2, "La": 3}[cation]
    phi_star = {"Na": 0.75, "Ca": 0.4, "La": 0.0001}[cation]
    diafiltrate_flow = 2.5 if cation == "La" else 5

    Dm_Cl = 2.03  # um2/s
    Dm_over_l_value = 60  # um/s
    l_m = (Dm_Cl / Dm_over_l_value) / 1e6  # m

    default_args = (
        ["Cl"],
        {"feed": 17, "diafiltrate": [1e-10, 1e-10, 1e-10, diafiltrate_flow, 1e-10]},
        True,
    )
    model = build_stage_model(
        cation_list=[cation],
        inlet_concentration={
            "feed": {
                cation: concentration,
                "Cl": chloride_multiplier * concentration,
            },
            "diafiltrate": {
                cation: [1e-10, 1e-10, 1e-10, 10, 1e-10],
                "Cl": [1e-10, 1e-10, 1e-10, chloride_multiplier * 10, 1e-10],
            },
        },
        default_args=default_args,
        NFE_args=[10, 10, 10],
        applied_pressure=pressure,
        pressure_drop=0.5,
        data_membrane_thickness=l_m,
        non_Donnan_partition_dict={cation: phi_star, "Cl": 0.05},
        num_tubes=num_tubes,
        initialize_and_solve=True,
    )
    return get_stage_tube_conditions(
        model, cation, chloride_multiplier, concentration, pressure, num_tubes
    )


def main():
    scan = SCAN
    if len(sys.argv) == 4:
        scan = {
            sys.argv[1]: {
                "concentrations": [float(c) for c in sys.argv[2].split(",")],
                "pressures": [float(p) for p in sys.argv[3].split(",")],
            }
        }
    elif len(sys.argv) != 1:
        sys.exit("usage: scan_stage_feasibility.py [cation concs pressures]")

    new_file = not os.path.exists(OUTFILE)
    with open(OUTFILE, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if new_file:
            writer.writeheader()

        for cation, cfg in scan.items():
            for concentration in cfg["concentrations"]:
                for pressure in cfg["pressures"]:
                    print(f"\n##### SCAN {cation}: P = {pressure}, c = {concentration}")
                    row = {
                        "cation": cation,
                        "tube_1_pressure": pressure,
                        "tube_1_feed_conc": concentration,
                    }
                    try:
                        tubes = scan_case(cation, concentration, pressure)
                        row["status"] = "solved"
                        row["min_tube_retentate_flow"] = min(
                            t["tube_retentate_flow"] for t in tubes
                        )
                        row["min_tube_inlet_flow"] = min(
                            t["tube_inlet_flow"] for t in tubes
                        )
                        row["max_tube_inlet_conc"] = max(
                            t["tube_inlet_conc"] for t in tubes
                        )
                        row["tube_retentate_flows"] = " ".join(
                            f"{t['tube_retentate_flow']:.2f}" for t in tubes
                        )
                    except Exception as e:  # record and continue
                        traceback.print_exc()
                        row["status"] = "failed"
                        row["message"] = str(e).splitlines()[0][:120]
                    writer.writerow(row)
                    f.flush()


if __name__ == "__main__":
    main()
