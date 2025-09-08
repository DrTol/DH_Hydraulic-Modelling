from __future__ import annotations

# -*- coding: utf-8 -*-
"""
Developed on Thu Nov 12 14:44:31 2020
Updated on Sun Sep  7 05:07:46 2025 (support by ChatGPT)

@author: Dr. Hakan İbrahim Tol
"""

"""
GT4DH – Connectivity (incidence) & loop matrices + hydraulic helper functions.

Functions
- PressureLoss(L, D, mFlow, aRou): Darcy–Weisbach pressure drop [bar]
- PL_valve(mFlow, kV): valve pressure loss model [bar]
- returnMatrices(PipeList, EUList): builds A_simple and M (loop) matrices
"""

import time
import math
import numpy as np

# ---------------------------
# Hydraulic helper functions
# ---------------------------

def PressureLoss(L, D, mFlow, aRou):
    """Pressure loss in a circular pipe (SI units). Returns bar."""
    g = 9.80665         # m/s2
    rho = 977.74        # kg/m3 (≈ 70 °C)
    mu  = 0.0004024     # Pa·s  (≈ 70 °C)

    def reynolds(mFlow, D_mm):
        D_m = D_mm / 1000.0
        # mFlow is mass flow [kg/s]; volumetric flow = mFlow / rho
        # Mean velocity v = 4 * mFlow / (pi * D * rho)
        return 4.0 * mFlow / (math.pi * D_m * mu)

    def f_clamond(Re, K):
        # Explicit Darcy friction factor approximation
        X1 = K * Re * 0.123968186335417556
        X2 = math.log(Re) - 0.779397488455682028
        DWc = X2 - 0.2
        E = (math.log(X1 + DWc) - 0.2) / (1 + X1 + DWc)
        DWc = DWc - (1 + X1 + DWc + 0.5 * E) * E * (X1 + DWc) / (1 + X1 + DWc + E * (1 + E / 3))
        E = (math.log(X1 + DWc) + DWc - X2) / (1 + X1 + DWc)
        DWc = DWc - (1 + X1 + DWc + 0.5 * E) * E * (X1 + DWc) / (1 + X1 + DWc + E * (1 + E / 3))
        DWc = 1.151292546497022842 / DWc
        return DWc * DWc

    Re = reynolds(mFlow, D)
    if Re < 2300.0:
        f = 64.0 / Re
    else:
        f = f_clamond(Re, aRou / D)

    D_m = D / 1000.0
    # Δp [Pa] = 8 f L m^2 / (π^2 ρ^2 D^5) * g? (bring to head/pressure form)
    # Here: consistent with legacy implementation → convert to bar at the end
    dp_bar = (8.0 * f * L * (mFlow ** 2) /
              (math.pi ** 2 * (rho ** 2) * 9.80665 * (D_m ** 5))) / 10.1971621297792
    return dp_bar

def PL_valve(mFlow, kV):
    """Valve pressure loss model. Returns list of bar values aligned with mFlow."""
    rho = 977.74
    vFlow_m3ph = (mFlow / rho) * 3600.0
    return [(vF / kv) ** 2 * (rho / 1000.0) if kv != 0 else 0.0
            for vF, kv in zip(vFlow_m3ph, kV)]

# ---------------------------
# Matrix builder (GT4DH)
# ---------------------------

def returnMatrices(PipeList, EUList):
    """
    Build:
      A_simple : incidence/connectivity with service branches on supply side
      A_expanded : supply + return stacked
      M : loop/energy matrix (constructed from service-path logic)
    PipeList uses 1-based node IDs (as in the Excel files).
    """
    t0 = time.perf_counter()

    n_N = int(max(EUList))
    n_P = int(len(PipeList))

    A = np.zeros((n_N, n_P))
    A[0, 0] = 1  # source feeder orientation

    N_s = PipeList[:, 0] - 1
    N_e = PipeList[:, 1] - 1

    for i_p in range(1, n_P):
        A[N_s[i_p] - 1, i_p] = -1
        A[N_e[i_p] - 1, i_p] = +1

    # Service connections (supply side)
    n_S = len(EUList)
    A_sp_supply = np.zeros((n_N, n_S))
    for i_s in range(n_S):
        A_sp_supply[EUList[i_s] - 1, i_s] = -1

    # Return-side stacking
    A_source        = A[:, 0]
    A_source_return = -A_source
    A_supply        = A[:, 1:]
    A_return        = -A_supply
    A_sp_return     = -A_sp_supply

    A_top    = np.concatenate((A_source.reshape(-1, 1), A_supply, A_sp_supply, np.zeros((n_N, n_P - 1))), axis=1)
    A_bottom = np.concatenate((A_source_return.reshape(-1, 1), np.zeros((n_N, n_P - 1)), A_sp_return, A_return), axis=1)
    A_expanded = np.concatenate((A_top, A_bottom), axis=0)

    A_simple = np.concatenate((A, A_sp_supply), axis=1)

    is_visited = np.zeros((n_S, 1))
    M_simple   = np.zeros((n_S, n_S + n_P))

    def find_sp(A_simple_, i_sp, last_node):
        M_temp = np.zeros((1, A_simple_.shape[1]))
        pivot_node = int(np.where(A_simple_[:, n_P + i_sp] == -1)[0])
        while pivot_node != last_node:
            # find forward pipe (column with +1 at pivot_node)
            idx = np.where(A_simple_[pivot_node, :] == 1)[0]
            if idx.size == 0:
                break
            pre_pipe = int(idx)
            M_temp[0, pre_pipe] = 1
            # hop to previous node (row with -1 in that pipe column)
            pivot_node = int(np.where(A_simple_[:, pre_pipe] == -1)[0])
        return M_temp

    def find_parent(A_simple_, i_spN, i_spP):
        M_N = find_sp(A_simple_, i_spN, 0)
        M_P = find_sp(A_simple_, i_spP, 0)
        M_J = M_N * M_P
        if not np.any(M_J):
            return 0
        return int(np.where(M_J == 1)[0][-1])

    def find_innerLoop(A_simple_, i_sp):
        M_temp  = np.zeros((1, A_simple_.shape[1]))
        node_sN = int(np.where(A_simple_[:, n_P + i_sp]     == -1)[0])
        node_sP = int(np.where(A_simple_[:, n_P + i_sp - 1] == -1)[0])

        if node_sN == node_sP:
            M_temp[0, n_P + i_sp]     = 1
            M_temp[0, n_P + i_sp - 1] = -1
        else:
            parent_pipe  = find_parent(A_simple_, i_sp, i_sp - 1)
            junction_node = int(np.where(A_simple_[:, parent_pipe] == 1)[0])
            M_N = find_sp(A_simple_, i_sp,     junction_node)
            M_P = find_sp(A_simple_, i_sp - 1, junction_node)
            M_temp = M_N - M_P
            M_temp[0, n_P + i_sp]     = 1
            M_temp[0, n_P + i_sp - 1] = -1
        return M_temp

    for i_sp in range(n_S):
        if i_sp == 0:
            M_simple[0, 0] = 1
            last_node = 0
            while is_visited[i_sp] == 0:
                if A_simple[last_node, n_P + i_sp] == -1:
                    M_simple[i_sp, n_P + i_sp] = 1
                    is_visited[i_sp] = 1
                else:
                    M_simple[i_sp, :] = find_sp(A_simple, i_sp, last_node)
                    M_simple[i_sp, n_P + i_sp] = 1
                    is_visited[i_sp] = 1
        else:
            M_simple[i_sp, :] = find_innerLoop(A_simple, i_sp)

    M = np.concatenate((M_simple, M_simple[:, 1:n_P]), axis=1)
    M[0, 0] = 1

    t1 = time.perf_counter()
    print(f"Time Elapsed inside returnMatrices(): {t1 - t0:.3f} s")
    return A_simple, A_expanded, M
