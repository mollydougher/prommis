"""
Feasibility scan for a single membrane tube over an extended input box, used to
choose the grid for the tube surrogate training data. The stage data shows tubes
2-5 see inlet flows down to ~4.4 m3/h and cation concentrations up to ~267
mol/m3, outside the original tube training box (flow 8-17, concentration
50-200). For each (cation, feed flow, feed concentration, pressure) the tube is
built and solved from scratch; failures are recorded instead of stopping the
scan. Results are appended to tube_feasibility_scan.csv as they are produced.

Run from this directory:
    python scan_tube_feasibility.py                        # full default scan
    python scan_tube_feasibility.py Na 4,6 250,300 3,5     # one cation:
                                    # feed flows, concentrations, pressures

Author: Claude
"""

import csv
import os
import sys
import traceback

from pyomo.environ import value

from generate_synthetic_data import (
    build_tube_model,
    get_average_flux,
    get_average_water_flux,
)

OUTFILE = "tube_feasibility_scan.csv"

# feed flow (m3/h), cation concentration (mol/m3), applied pressure (bar)
FLOWS = [4, 6, 8, 17]
CONCENTRATIONS = [50, 200, 250, 300]
SCAN = {
    "Na": {"pressures": [3, 5, 8, 13]},
    "Ca": {"pressures": [5, 8, 10, 15]},
    "La": {"pressures": [7, 10, 12, 17]},
}

FIELDS = [
    "cation",
    "feed_flow",
    "feed_conc",
    "pressure",
    "status",
    "ion_flux",
    "water_flux",
    "retentate_flow",
    "permeate_flow",
    "message",
]


def scan_case(cation, feed_flow, concentration, pressure):
    # same tube settings as generate_single_salt_tube_data
    chloride_multiplier = {"Na": 1, "Ca": 2, "La": 3}[cation]
    phi_star = {"Na": 0.75, "Ca": 0.4, "La": 0.0001}[cation]

    Dm_Cl = 2.03  # um2/s
    Dm_over_l_value = 60  # um/s
    l_m = (Dm_Cl / Dm_over_l_value) / 1e6  # m

    model = build_tube_model(
        cation_list=[cation],
        inlet_concentration={
            "feed": {cation: concentration, "Cl": chloride_multiplier * concentration},
            "diafiltrate": {cation: 1e-10, "Cl": 1e-10},
        },
        default_args=(["Cl"], {"feed": feed_flow, "diafiltrate": 1e-10}, True),
        NFE_args=[10, 10, 10],
        applied_pressure=pressure,
        data_membrane_thickness=l_m,
        multiplier_H_feed=1.2,
        multiplier_H_perm=1,
        non_Donnan_partition_dict={cation: phi_star, "Cl": 0.05},
        initialize_and_solve=True,
    )

    blk = model.fs.membrane
    t = blk.time.first()
    x_out = blk.dimensionless_module_length.last()
    return {
        "ion_flux": get_average_flux(model, cation, stage_bool=False),
        "water_flux": get_average_water_flux(model, stage_bool=False),
        "retentate_flow": value(blk.retentate_flow_volume[t, x_out]),
        "permeate_flow": value(blk.permeate_outlet_flow[t]),
    }


def main():
    flows, concentrations, scan = FLOWS, CONCENTRATIONS, SCAN
    if len(sys.argv) == 5:
        flows = [float(x) for x in sys.argv[2].split(",")]
        concentrations = [float(x) for x in sys.argv[3].split(",")]
        scan = {sys.argv[1]: {"pressures": [float(x) for x in sys.argv[4].split(",")]}}
    elif len(sys.argv) != 1:
        sys.exit("usage: scan_tube_feasibility.py [cation flows concs pressures]")

    new_file = not os.path.exists(OUTFILE)
    with open(OUTFILE, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if new_file:
            writer.writeheader()

        for cation, cfg in scan.items():
            for feed_flow in flows:
                for concentration in concentrations:
                    for pressure in cfg["pressures"]:
                        print(
                            f"\n##### SCAN {cation}: flow = {feed_flow}, "
                            f"c = {concentration}, P = {pressure}"
                        )
                        row = {
                            "cation": cation,
                            "feed_flow": feed_flow,
                            "feed_conc": concentration,
                            "pressure": pressure,
                        }
                        try:
                            row.update(
                                scan_case(cation, feed_flow, concentration, pressure)
                            )
                            row["status"] = "solved"
                        except Exception as e:  # record and continue
                            traceback.print_exc()
                            row["status"] = "failed"
                            row["message"] = str(e).splitlines()[0][:120]
                        writer.writerow(row)
                        f.flush()


if __name__ == "__main__":
    main()
