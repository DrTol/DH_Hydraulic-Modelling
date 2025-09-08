# -*- coding: utf-8 -*-
"""
Developed on Tue Mar 23 14:20:18 2021
Updated on Sun Sep  7 05:02:35 2025 (support by ChatGPT)

@author: Dr. Hakan İbrahim Tol
"""

"""
Solver and hydraulics utilities.

Includes
- cMass(): conservation-of-mass expansion with square/least-squares solve
- f_ga(): GA objective (loop energy residual)
- f_nr(), J_nr(): Newton–Raphson residual and finite-difference Jacobian
- run_solvers(): orchestrates GA and NR, prints timing and reports pipe flows (|·|)
"""

import time
import numpy as np
from geneticalgorithm import geneticalgorithm as ga

from function_GT4DH import PressureLoss as PL_p
from function_GT4DH import PL_valve     as PL_v

def run_solvers(
    A_s, M, L_P, D_P, h_L, k_V, EUList, PipeList,
    print_summary=True,
    ga_params=None
):
    nP_main = PipeList.shape[0]
    nS      = len(EUList)

    def _unpack_mF(mF_full):
        mF_full = np.reshape(mF_full, -1)
        mF_pipes_supply = mF_full[:nP_main]
        mF_end_users    = mF_full[nP_main:nP_main+nS]
        mF_pipes_return = mF_full[nP_main+nS:]
        return mF_pipes_supply, mF_end_users, mF_pipes_return

    def _fmt(v):
        return np.round(v, 4)

    # Simple roughness map (legacy)
    aR = [0.01 if d < 30.0 else 0.1 for d in D_P]

    # ---------------------------
    # Conservation of Mass
    # ---------------------------
    def cMass(V, A_s_local):
        """
        Solve for unknown branch flows:
          A_unknown * x_unknown = (b - A_known * x_known).
        Square → exact solve; otherwise → least-squares.
        """
        b = np.zeros((A_s_local.shape[0], 1))
        x_all = np.zeros((A_s_local.shape[1], 1))
        x_all[-len(EUList):, 0] = V

        i_known   = np.nonzero(x_all.ravel())[0]
        i_unknown = np.where(x_all.ravel() == 0)[0]

        A_known = A_s_local[:, i_known]
        x_known = x_all[i_known, :]

        A_unknown = A_s_local[:, i_unknown]
        rhs = b - A_known @ x_known

        if A_unknown.shape[0] == A_unknown.shape[1]:
            x_unknown = np.linalg.solve(A_unknown, rhs)
        else:
            x_unknown, *_ = np.linalg.lstsq(A_unknown, rhs, rcond=None)

        # Full vector: [unknowns, knowns, return-side copy (from 2nd unknown onward)]
        return np.concatenate((x_unknown, x_known, x_unknown[1:]), axis=0)

    # ---------------------------
    # GA objective
    # ---------------------------
    def f_ga(x):
        mF = np.reshape(cMass(x, A_s), -1)
        dP_p = np.array(list(map(PL_p, L_P, D_P, mF, aR)))
        dP_v = PL_v(mF, k_V)
        return np.sum(np.abs(M @ (dP_p + dP_v - h_L)))

    # ---------------------------
    # NR residual and Jacobian
    # ---------------------------
    def f_nr(x):
        mF = np.reshape(cMass(x, A_s), -1)
        dP_p = np.array(list(map(PL_p, L_P, D_P, mF, aR)))
        dP_v = PL_v(mF, k_V)
        return M @ (dP_p + dP_v - h_L)

    def J_nr(f, x, dx=1e-8):
        n = len(x)
        Fx = f(x)
        J = np.zeros((n, n))
        for j in range(n):
            step = (abs(x[j]) * dx) if x[j] != 0 else dx
            xj = np.array(x, dtype=float)
            xj[j] += step
            J[:, j] = (f(xj) - Fx) / step
        return J

    results = {"GA": {}, "NR": {}}

    # -------------
    # Genetic Algorithm
    # -------------
    if print_summary:
        print("Optimization Running (Seat Your Belts)")

    t_ga0 = time.perf_counter()

    if ga_params is None:
        ga_params = {
            'max_num_iteration': 1e3,
            'population_size': 250,
            'mutation_probability': 0.5,
            'elit_ratio': 0.25,
            'crossover_probability': 0.25,
            'parents_portion': 0.5,
            'crossover_type': 'two_point',
            'max_iteration_without_improv': 25
        }

    varbound = np.array([[0.0, 1.0]] * nS)

    model = ga(function=f_ga,
               dimension=nS,
               variable_type='real',
               variable_boundaries=varbound,
               algorithm_parameters=ga_params)

    model.run()

    t_ga1 = time.perf_counter()
    if print_summary:
        print(f"Time Elapsed for GA: {t_ga1 - t_ga0:.3f} s")

    x_ga = np.asarray(model.output_dict['variable'], dtype=float)
    mF_ga_full = np.reshape(cMass(x_ga, A_s), -1)
    mF_ga_pipes, mF_ga_eu, mF_ga_ret = _unpack_mF(mF_ga_full)

    # Report pipe segments as magnitudes (display only)
    mF_ga_pipes_rep = np.abs(mF_ga_pipes)
    mF_ga_ret_rep   = np.abs(mF_ga_ret)

    results["GA"] = {
        "x": x_ga,
        "mF_full": mF_ga_full,
        "mF_pipes": mF_ga_pipes_rep,
        "mF_eu": mF_ga_eu,
        "mF_ret": mF_ga_ret_rep,
        "time": (t_ga1 - t_ga0)
    }

    if print_summary:
        print("GA: end-user flows (x)        :", _fmt(x_ga))
        print("GA: mF (end-users)            :", _fmt(mF_ga_eu))
        print("GA: mF (supply pipe segments) :", _fmt(mF_ga_pipes_rep))
        print("GA: mF (return pipe segments) :", _fmt(mF_ga_ret_rep))

    # -------------
    # Newton–Raphson
    # -------------
    if print_summary:
        print("Newton-Raphson in Run (Seat Your Belts)")

    t_nr0 = time.perf_counter()

    err = 10.0
    tol = 0.1
    x0  = np.array([2.0] * nS, dtype=float)

    while tol < err:
        F_o = f_nr(x0)
        J_o = J_nr(f_nr, x0)
        x   = np.abs(x0 - np.linalg.inv(J_o) @ F_o)
        F_n = f_nr(x)
        err = float(np.sum(np.abs(F_n - F_o)))
        if print_summary:
            print("error ", err)
        x0 = x

    t_nr1 = time.perf_counter()
    if print_summary:
        print(f"Time Elapsed for NR: {t_nr1 - t_nr0:.3f} s")

    x_nr = np.asarray(x0, dtype=float)
    mF_nr_full = np.reshape(cMass(x_nr, A_s), -1)
    mF_nr_pipes, mF_nr_eu, mF_nr_ret = _unpack_mF(mF_nr_full)

    mF_nr_pipes_rep = np.abs(mF_nr_pipes)
    mF_nr_ret_rep   = np.abs(mF_nr_ret)

    results["NR"] = {
        "x": x_nr,
        "mF_full": mF_nr_full,
        "mF_pipes": mF_nr_pipes_rep,
        "mF_eu": mF_nr_eu,
        "mF_ret": mF_nr_ret_rep,
        "time": (t_nr1 - t_nr0)
    }

    if print_summary:
        print("NR: end-user flows (x)        :", _fmt(x_nr))
        print("NR: mF (end-users)            :", _fmt(mF_nr_eu))
        print("NR: mF (supply pipe segments) :", _fmt(mF_nr_pipes_rep))
        print("NR: mF (return pipe segments) :", _fmt(mF_nr_ret_rep))

    return results
