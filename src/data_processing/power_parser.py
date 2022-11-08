import os
from posixpath import split
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix, diags


# from grids import *

import sys

import pandapower as pp
import pandapower.networks as pn
from pandapower.pypower.ppoption import ppoption
from pandapower.auxiliary import _init_runopp_options
from pandapower.auxiliary import ppException, _clean_up, _add_auxiliary_elements
from pandapower.results import (
    _copy_results_ppci_to_ppc,
    init_results,
    verify_results,
    _extract_results,
)
from pandapower.pd2ppc import _pd2ppc
from pandapower.pypower.add_userfcn import add_userfcn
from pandapower.pypower.ppoption import ppoption
from pandapower.pypower.opf import opf
from pandapower.pypower.idx_bus import VM
from pandapower.optimal_powerflow import _add_dcline_constraints, _run_pf_before_opf
from time import perf_counter
from numpy import zeros, c_, shape
from pandapower.pypower.idx_brch import MU_ANGMAX
from pandapower.pypower.idx_bus import MU_VMIN
from pandapower.pypower.idx_gen import MU_QMIN
from pandapower.pypower.idx_brch import F_BUS, T_BUS, RATE_A, BR_B
from pandapower.pypower.idx_bus import GS, BS
from pandapower.pypower.opf_setup import opf_setup
from pandapower.pypower.opf_args import opf_args2
from pandapower.opf.validate_opf_input import _check_necessary_opf_parameters
from pandapower.auxiliary import (
    _check_bus_index_and_print_warning_if_high,
    _check_gen_index_and_print_warning_if_high,
)

# try:
#     import pandaplan.core.pplog as logging
# except ImportError:
#     import logging
import logging
from pandapower.pypower.makeYbus import makeYbus, branch_vectors


logger = logging.getLogger(__name__)

sys.path.append("..")


def parse_str(line: str, delim: str = "\t") -> list:
    """
    Parses string of delimited numbers, removes semicolons

    Args:
        line (str): string to parse
        delim (str, optional): delimiter. Defaults to '\t'.

    Returns:
        list: parsed numbers in a format of list
    """
    splitted = line.split(delim)
    while "" in splitted:
        splitted.remove("")
    output = []
    for v in splitted:
        curr_el = v.replace(";", "")
        if "." in v:
            output.append(float(curr_el))
        else:
            output.append(int(curr_el))
    return output


def parse_m(case_path: str, path_to_dump: str = "data/") -> None:
    """Parse matpower file into pandas dataframes

    Args:
        case_path (str): where is the matpower case.m
        path_to_dump (str, optional): where to save parsing results. Defaults to "data/".
    """

    with open(case_path, "r") as f:
        lines = f.read().splitlines()
    if not os.path.exists(path_to_dump):
        os.makedirs(path_to_dump)
    flag_bus = False
    flag_gen = False
    flag_branch = False
    flag_gencost = False
    BusData = pd.DataFrame()
    GenData = pd.DataFrame()
    BranchData = pd.DataFrame()
    GenCostData = pd.DataFrame()
    output = {}
    for i in range(len(lines)):
        line = lines[i]
        if "mpc.baseMVA" in line:
            row_data = [v for v in line.split("= ")[1:]]
            row_data[-1] = np.array(row_data[-1][0:-1], dtype=float)
            print("Reading BaseMVA...")
            # with open(os.path.join(path_to_dump, "BaseMVA.csv"), "w") as fs:
            #     np.savetxt(fs, row_data)
            # print("Array exported to file")
            output["BaseMVA"] = row_data[0]
            # writedlm("data/" * "BaseMVA" * ".txt", row_data)

            # CSV.write("data/" * "BaseMVA" * ".csv", row_data)

        if "];" in line:
            print("Reading finished...")
            flag_bus = False
            flag_branch = False
            flag_gen = False
            flag_gencost = False

        if flag_bus:
            row_data = parse_str(line, delim="\t")
            BusData.loc[len(BusData.index)] = row_data

        if flag_gen:
            row_data = parse_str(line, delim="\t")
            GenData.loc[len(GenData.index)] = row_data

        if flag_branch:
            row_data = parse_str(line, delim="\t")
            BranchData.loc[len(BranchData.index)] = row_data

        if flag_gencost:
            row_data = parse_str(line, delim="\t")
            GenCostData.loc[len(GenCostData.index)] = row_data

        if "%% bus data" in line:
            cols_read = lines[i + 1].split("\t")[1:]
            BusData[cols_read] = None

            print("Reading BusData...")
            print("Number of columns: ", len(BusData.columns))

        if "mpc.bus " in line:
            flag_bus = True

        if "%% generator data" in line:
            cols_read = lines[i + 1].split("\t")[1:]
            GenData[cols_read] = None

            print("Reading GenData...")
            print("Number of columns: ", len(GenData.columns))

        if "mpc.gen " in line:
            flag_gen = True

        if "%% branch data" in line:
            cols_read = lines[i + 1].split("\t")[1:]
            BranchData[cols_read] = None

            print("Reading BranchData...")
            print("Number of columns: ", len(BranchData.columns))

        if "mpc.branch " in line:
            flag_branch = True

        if "%% generator cost data" in line:
            cols_gencost = []
            for col in lines[i + 2].split("\t")[1:]:
                if (not "c" in col) and (not "..." in col):
                    cols_gencost.append(col)
            GenCostData[cols_gencost] = None

            row_ahead = parse_str(lines[i + 4], delim="\t")
            n_coeffs = len(row_ahead) - len(GenCostData.columns)
            new_cols = []
            for n_ in range(n_coeffs):
                new_cols.append("c" + str(n_coeffs - n_))
            GenCostData[new_cols] = None

            print("Reading GenCostData...")
            print("Number of columns: ", len(GenCostData.columns))

        if "mpc.gencost " in line:
            flag_gencost = True

    DFs = [BusData, GenData, BranchData, GenCostData]
    DFs_names = ["bus", "gen", "branch", "gencost"]

    for (df, name) in zip(DFs, DFs_names):
        output[name] = df
        # df.to_csv(os.path.join(path_to_dump, name + ".csv"))

    return output


def parse_pn(net):
    # try:
    #     net = grid_name_map[case_name]
    # except KeyError:
    #     print("Available case names are:")
    #     print(grid_name_map.keys())
    #     raise KeyError

    calculate_voltage_angles = True
    check_connectivity = (True,)
    suppress_warnings = True
    switch_rx_ratio = 2
    delta = 1e-10
    init = "flat"
    numba = (True,)
    trafo3w_losses = "hv"
    consider_line_temperature = False
    _check_necessary_opf_parameters(net, logger)
    _init_runopp_options(
        net,
        calculate_voltage_angles=calculate_voltage_angles,
        check_connectivity=check_connectivity,
        switch_rx_ratio=switch_rx_ratio,
        delta=delta,
        init=init,
        numba=numba,
        trafo3w_losses=trafo3w_losses,
        consider_line_temperature=consider_line_temperature,
    )
    _check_bus_index_and_print_warning_if_high(net)
    _check_gen_index_and_print_warning_if_high(net)

    ac = net["_options"]["ac"]
    init = net["_options"]["init"]
    ppopt = ppoption(PF_DC=not ac, INIT=init)

    net["OPF_converged"] = False
    net["converged"] = False
    _add_auxiliary_elements(net)

    if not ac or net["_options"]["init_results"]:
        verify_results(net)
    else:
        init_results(net, "opf")

    ppc, ppci = _pd2ppc(net)

    if not ac:
        ppci["bus"][:, VM] = 1.0
    net["_ppc_opf"] = ppci
    if len(net.dcline) > 0:
        ppci = add_userfcn(ppci, "formulation", _add_dcline_constraints, args=net)

    # if init == "pf":
    #     ppci = _run_pf_before_opf(net, ppci)
    # if suppress_warnings:
    #     with warnings.catch_warnings():
    #         warnings.simplefilter("ignore")
    #         result = opf(ppci, ppopt)
    # else:
    #     result = opf(ppci, ppopt)
    t0 = perf_counter()  ## start timer

    ## process input arguments
    ppc, ppopt = opf_args2(ppci, ppopt)

    ## add zero columns to bus, gen, branch for multipliers, etc if needed
    nb = shape(ppc["bus"])[0]  ## number of buses
    nl = shape(ppc["branch"])[0]  ## number of branches
    ng = shape(ppc["gen"])[0]  ## number of dispatchable injections
    if shape(ppc["bus"])[1] < MU_VMIN + 1:
        ppc["bus"] = c_[ppc["bus"], zeros((nb, MU_VMIN + 1 - shape(ppc["bus"])[1]))]

    if shape(ppc["gen"])[1] < MU_QMIN + 1:
        ppc["gen"] = c_[ppc["gen"], zeros((ng, MU_QMIN + 1 - shape(ppc["gen"])[1]))]

    if shape(ppc["branch"])[1] < MU_ANGMAX + 1:
        ppc["branch"] = c_[
            ppc["branch"], zeros((nl, MU_ANGMAX + 1 - shape(ppc["branch"])[1]))
        ]

    ##-----  convert to internal numbering, remove out-of-service stuff  -----
    # ppc = ext2int(ppc)

    ##-----  construct OPF model object  -----
    om = opf_setup(ppc, ppopt)

    ##-----  execute the OPF  -----

    ppc = om.get_ppc()
    baseMVA, bus, gen, branch, gencost = (
        ppc["baseMVA"],
        ppc["bus"],
        ppc["gen"],
        ppc["branch"],
        ppc["gencost"],
    )
    vv, _, nn, _ = om.get_idx()

    ## problem dimensions
    nb = bus.shape[0]  ## number of buses
    nl = branch.shape[0]  ## number of branches
    ny = om.getN("var", "y")  ## number of piece-wise linear costs

    ## linear constraints
    A, l, u = om.linear_constraints()

    ## bounds on optimization vars
    x0, xmin, xmax = om.getv()

    ## build admittance matrices
    Ybus, Yf, Yt, Ysh = makeYbus_custom(baseMVA, bus, branch)

    ## Build S matrix -- matrix of static line ratings

    row = branch[:, F_BUS].astype(np.int)
    col = branch[:, T_BUS].astype(np.int)
    data = branch[:, RATE_A].astype(np.float)
    S = csr_matrix((data, (row, col)), shape=(len(row), len(col)))
    S = S + S.T - diags(S.diagonal(), dtype=np.float)

    ## Build Shunt Susceptance ?????
    data = branch[:, BR_B].astype(np.float)
    bs = csr_matrix((data, (row, col)), shape=(len(row), len(col)))
    bs = bs + bs.T - diags(bs.diagonal(), dtype=np.float)
    # bs = bus[:, BS]
    return net, S, Ybus, Yf, Yt, bs, Ysh


def makeYbus_custom(baseMVA, bus, branch):
    """Builds the bus admittance matrix and branch admittance matrices.

    Returns the full bus admittance matrix (i.e. for all buses) and the
    matrices C{Yf} and C{Yt} which, when multiplied by a complex voltage
    vector, yield the vector currents injected into each line from the
    "from" and "to" buses respectively of each line. Does appropriate
    conversions to p.u.

    @see: L{makeSbus}

    @author: Ray Zimmerman (PSERC Cornell)
    @author: Richard Lincoln
    """
    ## constants
    nb = bus.shape[0]  ## number of buses
    nl = branch.shape[0]  ## number of lines

    ## for each branch, compute the elements of the branch admittance matrix where
    ##
    ##      | If |   | Yff  Yft |   | Vf |
    ##      |    | = |          | * |    |
    ##      | It |   | Ytf  Ytt |   | Vt |
    ##
    Ytt, Yff, Yft, Ytf = branch_vectors(branch, nl)
    ## compute shunt admittance
    ## if Psh is the real power consumed by the shunt at V = 1.0 p.u.
    ## and Qsh is the reactive power injected by the shunt at V = 1.0 p.u.
    ## then Psh - j Qsh = V * conj(Ysh * V) = conj(Ysh) = Gs - j Bs,
    ## i.e. Ysh = Psh + j Qsh, so ...
    ## vector of shunt admittances
    Ysh = (bus[:, GS] + 1j * bus[:, BS]) / baseMVA

    ## build connection matrices
    f = np.real(branch[:, F_BUS]).astype(int)  ## list of "from" buses
    t = np.real(branch[:, T_BUS]).astype(int)  ## list of "to" buses
    ## connection matrix for line & from buses
    Cf = csr_matrix((np.ones(nl), (range(nl), f)), (nl, nb))
    ## connection matrix for line & to buses
    Ct = csr_matrix((np.ones(nl), (range(nl), t)), (nl, nb))

    ## build Yf and Yt such that Yf * V is the vector of complex branch currents injected
    ## at each branch's "from" bus, and Yt is the same for the "to" bus end
    i = np.hstack([range(nl), range(nl)])  ## double set of row indices

    Yf = csr_matrix((np.hstack([Yff, Yft]), (i, np.hstack([f, t]))), (nl, nb))
    Yt = csr_matrix((np.hstack([Ytf, Ytt]), (i, np.hstack([f, t]))), (nl, nb))
    # Yf = spdiags(Yff, 0, nl, nl) * Cf + spdiags(Yft, 0, nl, nl) * Ct
    # Yt = spdiags(Ytf, 0, nl, nl) * Cf + spdiags(Ytt, 0, nl, nl) * Ct

    ## build Ybus
    Ybus = Cf.T * Yf + Ct.T * Yt + csr_matrix((Ysh, (range(nb), range(nb))), (nb, nb))
    Ybus.sort_indices()
    Ybus.eliminate_zeros()

    return Ybus, Yf, Yt, Ysh


# if __name__ == "main":
# case_name = "case9"
# case_path = os.path.join(
#     "/Users/aleks/Documents/study/phd/RAIC/power_sys/CNO/data",
#     case_name + ".m",
# )
# path_to_dump = os.path.join(
#     "/Users/aleks/Documents/study/phd/RAIC/power_sys/CNO/data/", case_name
# )
# output = parse_m(path_to_dump=path_to_dump, case_path=case_path)
