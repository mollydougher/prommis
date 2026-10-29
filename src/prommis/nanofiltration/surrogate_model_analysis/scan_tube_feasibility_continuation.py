"""
Continuation-based feasibility scan for a single membrane tube (default: La).

Fresh builds of the La tube are erratic, so (as in generate_single_salt_tube_data)
the tube is built once per pressure at a known-good point (feed flow 8 m3/h,
50 mol/m3) and every other condition is reached by continuation:
  1. the feed concentration is stepped up a ladder (steps of at most 25 mol/m3),
  2. at each concentration the feed flow is stepped down from 8 m3/h to the
     lowest flow of interest and back up before moving to the next concentration.
Each solve is checked with the same numerical-warning diagnostics the data
generator uses. If a flow excursion fails, the model is restored (or rebuilt and
walked back up the concentration ladder). Results are appended to
tube_feasibility_scan_continuation.csv as they are produced.

Run from this directory:
    python scan_tube_feasibility_continuation.py            # La, default grid
    python scan_tube_feasibility_continuation.py La 7,12    # cation, pressures

Author: Molly Dougher
"""

import csv
import os
import sys
import traceback

from pyomo.environ import value

from idaes.core.util.diagnostics_tools.diagnostics_toolbox import DiagnosticsToolbox

from generate_synthetic_data import (
    build_tube_model,
    get_average_flux,
    get_average_water_flux,
    solve_model,
)

OUTFILE = "tube_feasibility_scan_continuation.csv"

CATION = "La"
PRESSURES = [7, 10, 12, 17]
START_FLOW = 8  # m3/h, known to build fresh at the start concentration
START_CONC = 50  # mol/m3
CONCENTRATIONS = [50, 75, 100, 125, 150, 175, 200, 250, 300]
EXCURSION_FLOWS = [7, 6, 5, 4]  # descended in order, then climbed back to START_FLOW
MAX_CONC_STEP = 25

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

VALENCE = {"Na": 1, "Ca": 2, "La": 3}
PHI_STAR = {"Na": 0.75, "Ca": 0.4, "La": 0.0001}


def build_start(cation, pressure):
    Dm_Cl = 2.03  # um2/s
    Dm_over_l_value = 60  # um/s
    l_m = (Dm_Cl / Dm_over_l_value) / 1e6  # m
    return build_tube_model(
        cation_list=[cation],
        inlet_concentration={
            "feed": {cation: START_CONC, "Cl": VALENCE[cation] * START_CONC},
            "diafiltrate": {cation: 1e-10, "Cl": 1e-10},
        },
        default_args=(["Cl"], {"feed": START_FLOW, "diafiltrate": 1e-10}, True),
        NFE_args=[10, 10, 10],
        applied_pressure=pressure,
        data_membrane_thickness=l_m,
        multiplier_H_feed=1.2,
        multiplier_H_perm=1,
        non_Donnan_partition_dict={cation: PHI_STAR[cation], "Cl": 0.05},
        initialize_and_solve=True,
    )


def set_and_solve(model, cation, flow=None, conc=None):
    blk = model.fs.membrane
    if flow is not None:
        blk.feed_flow_volume.fix(flow)
    if conc is not None:
        for t in blk.time:
            blk.feed_conc_mol_comp[t, cation].fix(conc)
            blk.feed_conc_mol_comp[t, "Cl"].fix(VALENCE[cation] * conc)
        if conc >= 200:  # same scaling choice as the tube data generator for La
            blk.scaling_factor[blk.molar_ion_flux] = 1e3
    solve_model(model)
    DiagnosticsToolbox(model).assert_no_numerical_warnings(model)


def step_conc(model, cation, from_conc, to_conc, flow):
    n_steps = max(1, -(-abs(to_conc - from_conc) // MAX_CONC_STEP))
    for k in range(1, int(n_steps) + 1):
        set_and_solve(
            model,
            cation,
            flow=flow,
            conc=from_conc + (to_conc - from_conc) * k / n_steps,
        )


def results(model, cation):
    blk = model.fs.membrane
    t = blk.time.first()
    x_out = blk.dimensionless_module_length.last()
    return {
        "ion_flux": get_average_flux(model, cation, stage_bool=False),
        "water_flux": get_average_water_flux(model, stage_bool=False),
        "retentate_flow": value(blk.retentate_flow_volume[t, x_out]),
        "permeate_flow": value(blk.permeate_outlet_flow[t]),
    }


def scan_pressure(cation, pressure, writer, outfile):
    def record(flow, conc, status, model=None, message=""):
        row = {
            "cation": cation,
            "feed_flow": flow,
            "feed_conc": conc,
            "pressure": pressure,
            "status": status,
            "message": message,
        }
        if model is not None:
            row.update(results(model, cation))
        writer.writerow(row)
        outfile.flush()

    def rebuild(to_conc):
        """Fresh build at the start point, then climb the ladder to to_conc."""
        model = build_start(cation, pressure)
        step_conc(model, cation, START_CONC, to_conc, START_FLOW)
        return model

    print(f"\n##### SCAN {cation}: P = {pressure} (fresh build at start point)")
    try:
        model = build_start(cation, pressure)
    except Exception as e:
        traceback.print_exc()
        record(START_FLOW, START_CONC, "failed", message=str(e).splitlines()[0][:120])
        return

    conc = START_CONC
    for target in CONCENTRATIONS:
        print(f"\n##### SCAN {cation}: P = {pressure}, c = {target}, flow = {START_FLOW}")
        try:
            if target != conc:
                step_conc(model, cation, conc, target, START_FLOW)
            conc = target
            record(START_FLOW, conc, "solved", model)
        except Exception as e:
            traceback.print_exc()
            record(START_FLOW, target, "failed", message=str(e).splitlines()[0][:120])
            return  # higher concentrations were not reached

        # flow excursion at this concentration
        reached_all = True
        for flow in EXCURSION_FLOWS:
            print(f"\n##### SCAN {cation}: P = {pressure}, c = {conc}, flow = {flow}")
            try:
                set_and_solve(model, cation, flow=flow)
                record(flow, conc, "solved", model)
            except Exception as e:
                traceback.print_exc()
                record(flow, conc, "failed", message=str(e).splitlines()[0][:120])
                reached_all = False
                break
        if reached_all:
            for flow in EXCURSION_FLOWS[-2::-1] + [START_FLOW]:
                try:
                    set_and_solve(model, cation, flow=flow)
                except Exception:
                    reached_all = False
                    break
        if not reached_all:
            # restore the model at the start flow, or rebuild it
            try:
                set_and_solve(model, cation, flow=START_FLOW)
            except Exception:
                try:
                    model = rebuild(conc)
                except Exception as e:
                    traceback.print_exc()
                    record(
                        START_FLOW,
                        conc,
                        "failed",
                        message="could not restore model: " + str(e).splitlines()[0][:80],
                    )
                    return


def main():
    cation, pressures = CATION, PRESSURES
    if len(sys.argv) == 3:
        cation = sys.argv[1]
        pressures = [float(p) for p in sys.argv[2].split(",")]
    elif len(sys.argv) != 1:
        sys.exit("usage: scan_tube_feasibility_continuation.py [cation pressures]")

    new_file = not os.path.exists(OUTFILE)
    with open(OUTFILE, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if new_file:
            writer.writeheader()
        for pressure in pressures:
            scan_pressure(cation, pressure, writer, f)


if __name__ == "__main__":
    main()
