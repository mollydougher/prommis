"""
High-fidelity 5-tube stage test data for comparing the surrogate flowsheets.

The test conditions are tube 1 feed concentrations and applied pressures that lie
inside, but are not on, the training grid of the stage surrogate. For each cation
the stage is built at the lowest corner of the training grid (fresh builds are
erratic) and every test point is reached by continuation. Points already in
test_data/ are kept and only missing points are solved. The stage-average cation flux and
the per-tube conditions are saved in test_data/, in the same format as the
training data, along with the wall-clock time of the warm-started solve.

Run from this directory:  python generate_stage_test_data.py

Author: Claude
"""

import os
import time

import numpy as np

from pandas import DataFrame, read_csv

from generate_synthetic_data import (
    build_stage_model,
    fix_stage_diafiltrate_conc,
    get_average_flux,
    get_stage_tube_conditions,
    set_stage_pressure,
    solve_model,
    step_stage_feed_conc,
)

NUM_TUBES = 5
PRESSURE_DROP = 0.5  # bar per tube

# ranges of the stage training data (tube 1 feed concentration, tube 1 pressure)
STAGE_SETTINGS = {
    "Na": {
        "valence": 1,
        "phi_star": 0.75,
        "concentration": (150, 200),
        "pressure": (5, 7),
        "diafiltrate_flow": 5,
    },
    "Ca": {
        "valence": 2,
        "phi_star": 0.4,
        "concentration": (150, 200),
        "pressure": (8, 12),
        "diafiltrate_flow": 5,
    },
    "La": {
        "valence": 3,
        "phi_star": 0.0001,
        "concentration": (50, 100),
        "pressure": (12, 16),
        "diafiltrate_flow": 2.5,
    },
}

# test points as fractions of the (concentration, pressure) training ranges; the
# training grids sit at fractions 0, 0.5, and 1, so none of these are training points
TEST_FRACTIONS = [(0.25, 0.25), (0.75, 0.25), (0.25, 0.75), (0.75, 0.75), (0.5, 0.4), (0.4, 0.6)]

OUTPUT_FOLDER = "test_data/stage/single_salt"


def test_points(cation):
    cfg = STAGE_SETTINGS[cation]
    c_lo, c_hi = cfg["concentration"]
    p_lo, p_hi = cfg["pressure"]
    points = [
        (round(c_lo + f_c * (c_hi - c_lo), 4), round(p_lo + f_p * (p_hi - p_lo), 4))
        for f_c, f_p in TEST_FRACTIONS
    ]
    return sorted(points)  # neighboring points are close, which helps continuation


def diafiltrate_conc(cation, valence):
    return {
        cation: [1e-10, 1e-10, 1e-10, 10, 1e-10],
        "Cl": [1e-10, 1e-10, 1e-10, valence * 10, 1e-10],
    }


def build_stage(cation, concentration, pressure):
    cfg = STAGE_SETTINGS[cation]
    valence = cfg["valence"]

    Dm_Cl = 2.03  # um2/s
    Dm_over_l_value = 60  # um/s
    l_m = (Dm_Cl / Dm_over_l_value) / 1e6  # m

    default_args = (
        ["Cl"],
        {
            "feed": 17,
            "diafiltrate": [1e-10, 1e-10, 1e-10, cfg["diafiltrate_flow"], 1e-10],
        },
        True,
    )
    return build_stage_model(
        cation_list=[cation],
        inlet_concentration={
            "feed": {cation: concentration, "Cl": valence * concentration},
            "diafiltrate": diafiltrate_conc(cation, valence),
        },
        default_args=default_args,
        NFE_args=[10, 10, 10],
        applied_pressure=pressure,
        pressure_drop=PRESSURE_DROP,
        data_membrane_thickness=l_m,
        non_Donnan_partition_dict={cation: cfg["phi_star"], "Cl": 0.05},
        num_tubes=NUM_TUBES,
        initialize_and_solve=True,
    )


def move_stage(model, cation, from_point, to_point, max_pressure_step=1.0):
    """Continuation from one converged (concentration, pressure) to another."""
    valence = STAGE_SETTINGS[cation]["valence"]
    from_conc, from_pressure = from_point
    to_conc, to_pressure = to_point

    if to_conc != from_conc:
        step_stage_feed_conc(model, cation, valence, from_conc, to_conc)

    n_steps = max(1, int(np.ceil(abs(to_pressure - from_pressure) / max_pressure_step)))
    for k in range(1, n_steps + 1):
        set_stage_pressure(
            model,
            from_pressure + (to_pressure - from_pressure) * k / n_steps,
            PRESSURE_DROP,
            NUM_TUBES,
        )
        solve_model(model)
    fix_stage_diafiltrate_conc(model, diafiltrate_conc(cation, valence), NUM_TUBES)
    solve_model(model)


def main():
    stage_file = f"{OUTPUT_FOLDER}/mixed_chlorides"
    tube_file = f"{OUTPUT_FOLDER}/mixed_chlorides_tube_conditions"

    # resume: keep points that are already in the data and solve only the missing ones
    stage_rows = []
    tube_rows = []
    done = set()
    if os.path.exists(stage_file) and os.path.exists(tube_file):
        existing_stage = read_csv(stage_file)
        stage_rows = existing_stage.to_dict("records")
        tube_rows = read_csv(tube_file).to_dict("records")
        done = {
            (int(r["valency"]), round(r["feed_conc"], 4), round(r["pressure"], 4))
            for r in stage_rows
        }
        print(f"Found {len(done)} existing test points; solving only the missing ones")

    failed_cases = []

    for cation, cfg in STAGE_SETTINGS.items():
        valence = cfg["valence"]
        points = [
            point
            for point in test_points(cation)
            if (valence, round(point[0], 4), round(point[1], 4)) not in done
        ]
        # fresh builds are erratic, so the stage is built at the lowest corner of
        # the training grid, which is known to solve, and every test point is
        # reached by continuation from there (or from the previous point)
        anchor = (cfg["concentration"][0], cfg["pressure"][0])
        model = None
        current = None

        for concentration, pressure in points:
            print("=====================")
            print(f"{cation} test point: pressure = {pressure}, concentration = {concentration}")
            print("=====================")
            try:
                if model is None:
                    model = build_stage(cation, *anchor)
                    current = anchor

                start = time.perf_counter()
                move_stage(model, cation, current, (concentration, pressure))
                solve_time = time.perf_counter() - start

                flux = get_average_flux(model, cation, stage_bool=True, num_tubes=NUM_TUBES)
                tubes = get_stage_tube_conditions(
                    model, cation, valence, concentration, pressure, NUM_TUBES
                )
            except Exception as e:
                print(f"FAILED: {cation} P = {pressure}, c = {concentration}")
                failed_cases.append(
                    {
                        "valency": valence,
                        "feed_conc": concentration,
                        "pressure": pressure,
                        "message": str(e).splitlines()[0][:120],
                    }
                )
                model = None  # rebuild at the anchor for the next point
                current = None
                continue

            current = (concentration, pressure)
            stage_rows.append(
                {
                    "valency": valence,
                    "feed_conc": concentration,
                    "pressure": pressure,
                    "ion_flux": flux,
                    # wall-clock time of the warm-started continuation from the
                    # previous point (excludes the build at the anchor)
                    "solve_time_s": solve_time,
                    # 1 if the time includes a fresh build of the stage (points
                    # from earlier versions of this script); not a warm solve
                    "from_scratch": 0,
                }
            )
            tube_rows.extend(tubes)

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    DataFrame(stage_rows).to_csv(stage_file, index=False)
    DataFrame(tube_rows).to_csv(tube_file, index=False)

    failed_file = f"{OUTPUT_FOLDER}/mixed_chlorides_failed_cases"
    if failed_cases:
        print("The following test points did not solve and are not in the data:")
        for case in failed_cases:
            print(case)
        DataFrame(failed_cases).to_csv(failed_file, index=False)
    elif os.path.exists(failed_file):
        os.remove(failed_file)


if __name__ == "__main__":
    main()
