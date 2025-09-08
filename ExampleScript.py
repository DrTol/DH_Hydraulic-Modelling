# -*- coding: utf-8 -*-
"""
Developed on Tue Mar 23 14:20:18 2021
Updated on Sun Sep  7 05:02:35 2025 (support by ChatGPT)

@author: Dr. Hakan İbrahim Tol
"""

"""
HYDRAULIC ANALYSIS – Driver script

Workflow
  1) Read network data (Excel)
  2) Build incidence (A) and loop (M) matrices via GT4DH.returnMatrices()
  3) Solve end-user flows with GA and Newton–Raphson (NR)
  4) Expand to full pipe-flow vector via mass conservation
  5) Print flows and timing information
"""

import time
import numpy as np
import pandas as pd

from function_GT4DH import returnMatrices
from function_solver import run_solvers

# -------------------------
# USER SETTING: INPUT FILE
# -------------------------
# INPUT_XLSX = "00_NetworkData.xlsx"
INPUT_XLSX = "00_SimpleNetwork.xlsx"

# -------------------------
# 1) READ INPUT SHEETS
# -------------------------
t0 = time.perf_counter()
print("Reading Network Input Data from:", INPUT_XLSX)

input_pipe = pd.read_excel(INPUT_XLSX, sheet_name="PipeList")
input_endU = pd.read_excel(INPUT_XLSX, sheet_name="EUList")

# Pipe data (node IDs remain 1-based as in Excel)
PipeList = input_pipe[["Predecessor Node", "Successor Node"]].astype(int).to_numpy()
L_P      = np.asarray(input_pipe["Length"].to_numpy(), dtype=float).reshape(-1)
D_P      = np.asarray(input_pipe["Pipe Diameter"].to_numpy(), dtype=float).reshape(-1)
h_L      = np.asarray(input_pipe["Pump Head Lift"].to_numpy(), dtype=float).reshape(-1)

# End-user data (keep 1-based node indexing)
EUList = np.asarray(input_endU["Node in Connection"].astype(int).to_numpy()).reshape(-1)
kV     = np.asarray(input_endU["kV"].to_numpy(), dtype=float).reshape(-1)

# Service pipe constants (same as the legacy example)
L_eu = 15.0   # [m]
D_eu = 20.0   # [mm]

t1 = time.perf_counter()
print(f"Time Elapsed for data load: {t1 - t0:.3f} s")

# --------------------------------
# 2) BUILD EXTENDED INPUT VECTORS
# --------------------------------
L_P_ext = np.concatenate((L_P, [L_eu]*len(EUList), L_P[1:]), axis=0)
D_P_ext = np.concatenate((D_P, [D_eu]*len(EUList), D_P[1:]), axis=0)
h_L_ext = np.concatenate((h_L, [0.0]*len(EUList),  h_L[1:]), axis=0)
k_V_ext = np.concatenate(([0.0]*len(PipeList), kV, [0.0]*(len(PipeList)-1)), axis=0)

# ------------------------------
# 3) MATRICES (GT4DH)
# ------------------------------
print("Incidence & Loop Matrix Construction (GT4DH)")
t2 = time.perf_counter()
A_s, A_expanded, M = returnMatrices(PipeList, EUList)  # expects 1-based indices
t3 = time.perf_counter()
print(f"Time Elapsed for GT4DH.returnMatrices: {t3 - t2:.3f} s")

# ------------------------------
# 4) RUN SOLVERS (GA + NR)
# ------------------------------
results = run_solvers(
    A_s=A_s, M=M,
    L_P=L_P_ext, D_P=D_P_ext, h_L=h_L_ext, k_V=k_V_ext,
    EUList=EUList, PipeList=PipeList,
    print_summary=True
)

t4 = time.perf_counter()
print(f"Total runtime: {t4 - t0:.3f} s")

# Programmatic access example:
# results['GA']['x'], results['GA']['mF_pipes'], results['GA']['mF_eu'], results['GA']['mF_ret']
# results['NR']['x'], results['NR']['mF_pipes'], results['NR']['mF_eu'], results['NR']['mF_ret']
