from pyomo.environ import *
import numpy as np
import os
import sys
import pandas as pd
import pandapower as pp
import pandapower.networks as pn
from scipy import sparse

sys.path.append("/Users/aleks/Documents/study/phd/RAIC/power_sys/CNO")
import src.data_processing.power_parser as parser


def get_params_for_model(net):
    net, S, Ybus, Yf, Yt, bs, Ysh = parser.parse_pn(net)
    G = np.real(Ybus)
    B = np.imag(Ybus)

    # sources = ['sun', 'wind', 'voodoo']
    # time_periods = list(range(4))
    # sources = ['sun', 'wind', 'voodoo']
    # time_periods = list(range(4))
    n_gens = len(net["gen"]) + len(net["ext_grid"])
    n_buses = len(net["bus"])
    n_branches = S.shape[0]
    gen_idxs = list(
        np.hstack((net["gen"]["bus"].values, net["ext_grid"]["bus"].values))
    )  # list(range(n_gens))
    bus_idxs = list(range(n_buses))
    branch_idxs = list(dict(S.todok().items()).keys())
    return (
        net,
        bs,
        S,
        G,
        B,
        n_gens,
        n_buses,
        n_branches,
        gen_idxs,
        bus_idxs,
        branch_idxs,
    )


def spawn_ac_opf_model(
    net, bs, S, G, B, n_gens, n_buses, n_branches, gen_idxs, bus_idxs, branch_idxs
):
    # net, S, Ybus, Yf, Yt, bs, Ysh = parser.parse_pn(net)
    # G = np.real(Ybus)
    # B = np.imag(Ybus)

    # n_gens = len(net["gen"]) + len(net["ext_grid"])
    # n_buses = len(net["bus"])
    # n_branches = S.shape[0]
    # gen_idxs = list(
    #     np.hstack((net["gen"]["bus"].values, net["ext_grid"]["bus"].values))
    # )  # list(range(n_gens))
    # bus_idxs = list(range(n_buses))
    # branch_idxs = list(dict(S.todok().items()).keys())

    m = ConcreteModel()
    # sets
    m.gen_idxs = Set(initialize=gen_idxs)
    m.bus_idxs = Set(initialize=bus_idxs)
    m.branch_idxs = Set(initialize=branch_idxs)

    def Vm_bounds_rule(m, i):
        return (net["bus"].loc[i]["min_vm_pu"], net["bus"].loc[i]["max_vm_pu"])

    # def Va_bounds_rule(m, i):
    #    return (-np.pi, np.pi)
    # model.PriceToCharge = Var(model.A, domain=PositiveIntegers, bounds=fb)

    # Variables
    ## Uni commitment
    m.UC = Var(m.gen_idxs, within=Binary, initialize={gi: 0 for gi in m.gen_idxs})
    ## Voltages related variables
    m.Vm = Var(
        m.bus_idxs,
        domain=Reals,
        bounds=Vm_bounds_rule,
        initialize=np.random.randn(len(m.bus_idxs)),
    )
    m.Va = Var(
        m.bus_idxs,
        domain=Reals,
        initialize={
            idx: np.random.uniform(
                net["bus"].loc[idx]["min_vm_pu"], net["bus"].loc[idx]["max_vm_pu"]
            )
            for idx in m.bus_idxs
        },
    )

    ## Power generations

    m.Pg = Var(
        m.gen_idxs,
        domain=Reals,
        # bounds=Pg_bounds_rule,
        initialize={
            # idx: np.random.uniform(*Pg_bounds_rule(m, idx)) for idx in m.gen_idxs
            idx: np.random.randn() * net["sn_mva"]
            for idx in m.gen_idxs
        },
    )
    m.Qg = Var(
        m.gen_idxs,
        domain=Reals,
        # bounds=Qg_bounds_rule,
        initialize={
            # idx: np.random.uniform(*Qg_bounds_rule(m, idx)) for idx in m.gen_idxs
            idx: np.random.randn() * net["sn_mva"]
            for idx in m.gen_idxs
        },
    )

    ## Power flows through lines
    # def Flow_bounds(m, i):
    #    return (curr_gen['min_p_mw'], curr_gen['max_p_mw'])
    m.Pij = Var(
        m.branch_idxs,
        domain=Reals,
        initialize={
            idx: 1 * np.random.randn() * net["sn_mva"] for idx in m.branch_idxs
        },
    )
    m.Qij = Var(
        m.branch_idxs,
        domain=Reals,
        initialize={
            idx: 1 * np.random.randn() * net["sn_mva"] for idx in m.branch_idxs
        },
    )

    # m.Pij.value = np.random.randn(len(m.Pij)) * net['sn_mva']
    # m.Qij.value = np.random.randn(len(m.Qij)) * net['sn_mva']
    # Constraints
    ## UC
    def UC_rule(m, i):
        return m.UC[i] * (m.UC[i] - 1) == 0.0

    # m.UC_binary = Constraint(m.gen_idxs, rule=UC_rule)

    def Pg_bounds_rule_upper(m, i):
        # if i == 0:
        #     return Constraint.Skip
        if i in net["gen"]["bus"].values:
            curr_gen = net["gen"][net["gen"]["bus"] == i]
            return m.Pg[i] - curr_gen["max_p_mw"].values[0] * m.UC[i] <= 0.0

        if i in net["ext_grid"]["bus"].values:
            curr_gen = net["ext_grid"][net["ext_grid"]["bus"] == i]
            return m.Pg[i] - curr_gen["max_p_mw"].values[0] * m.UC[i] <= 0.0

    def Qg_bounds_rule_upper(m, i):
        # if i == 0:
        #     return Constraint.Skip
        if i in net["gen"]["bus"].values:
            curr_gen = net["gen"][net["gen"]["bus"] == i]
            return m.Qg[i] - curr_gen["max_q_mvar"].values[0] * m.UC[i] <= 0.0
        if i in net["ext_grid"]["bus"].values:
            curr_gen = net["ext_grid"][net["ext_grid"]["bus"] == i]
            return m.Qg[i] - curr_gen["max_q_mvar"].values[0] * m.UC[i] <= 0.0

    def Pg_bounds_rule_lower(m, i):
        # if i == 0:
        #     return Constraint.Skip
        if i in net["gen"]["bus"].values:
            curr_gen = net["gen"][net["gen"]["bus"] == i]
            return -m.Pg[i] + curr_gen["min_p_mw"].values[0] * m.UC[i] <= 0.0

        if i in net["ext_grid"]["bus"].values:
            curr_gen = net["ext_grid"][net["ext_grid"]["bus"] == i]
            return curr_gen["min_p_mw"].values[0] * m.UC[i] - m.Pg[i] <= 0.0

    def Qg_bounds_rule_lower(m, i):
        # if i == 0:
        #     return Constraint.Skip
        if i in net["gen"]["bus"].values:
            curr_gen = net["gen"][net["gen"]["bus"] == i]
            return curr_gen["min_q_mvar"].values[0] * m.UC[i] - m.Qg[i] <= 0.0
        if i in net["ext_grid"]["bus"].values:
            curr_gen = net["ext_grid"][net["ext_grid"]["bus"] == i]
            return curr_gen["min_q_mvar"].values[0] * m.UC[i] - m.Qg[i] <= 0.0

    m.Pg_upper = Constraint(m.gen_idxs, rule=Pg_bounds_rule_upper)
    m.Pg_lower = Constraint(m.gen_idxs, rule=Pg_bounds_rule_lower)
    m.Qg_lower = Constraint(m.gen_idxs, rule=Qg_bounds_rule_lower)
    m.Qg_upper = Constraint(m.gen_idxs, rule=Qg_bounds_rule_upper)
    ## Slack bus phase
    def slack_bus_rule_a(m, i):
        if i in net["ext_grid"]["bus"].values:
            return m.Va[i] == 0.0
        else:
            return Constraint.Skip

    def slack_bus_rule_m(m, i):
        if i in net["ext_grid"]["bus"].values:
            return m.Vm[i] == 1.0
        else:
            return Constraint.Skip

    m.slack_bus_a = Constraint(m.bus_idxs, rule=slack_bus_rule_a)
    # m.slack_bus_m = Constraint(m.bus_idxs, rule=slack_bus_rule_m)

    ## Power flow through lines
    def Pij_rule(m, i, j):
        # print(branch_idx)
        # i, j = branch_idx

        return m.Pij[i, j] == net.sn_mva * (
            m.Vm[i]
            * m.Vm[j]
            * (G[i, j] * cos(m.Va[i] - m.Va[j]) + B[i, j] * sin(m.Va[i] - m.Va[j]))
            - G[i, j] * m.Vm[i] ** 2
        )

    def Qij_rule(m, i, j):
        # i, j = branch_idx
        return m.Qij[i, j] == net.sn_mva * (
            m.Vm[i]
            * m.Vm[j]
            * (G[i, j] * sin(m.Va[i] - m.Va[j]) - B[i, j] * cos(m.Va[i] - m.Va[j]))
            + (B[i, j] - bs[i, j] / 2) * m.Vm[i] ** 2
        )

    m.power_flow_active = Constraint(m.branch_idxs, rule=Pij_rule)
    m.power_flow_reactive = Constraint(m.branch_idxs, rule=Qij_rule)

    ## Nodal balance
    def nodal_active_rule(m, bus_idx):
        current_load = net["load"][net["load"]["bus"] == bus_idx]["p_mw"].values
        if len(current_load) > 0:
            current_load = current_load[0]
            return (
                sum([m.Pg[g] for g in m.gen_idxs if g == bus_idx])
                - current_load
                - sum(
                    [m.Pij[br_idx] for br_idx in m.branch_idxs if br_idx[0] == bus_idx]
                )
                == 0.0
            )
        else:
            # return Constraint.Skip
            return (
                sum([m.Pg[g] for g in m.gen_idxs if g == bus_idx])
                - 0.0
                - sum(
                    [m.Pij[br_idx] for br_idx in m.branch_idxs if br_idx[0] == bus_idx]
                )
                == 0.0
            )

    def nodal_reactive_rule(m, bus_idx):
        current_load = net["load"][net["load"]["bus"] == bus_idx]["q_mvar"].values
        if len(current_load) > 0:
            current_load = current_load[0]
            return (
                sum([m.Qg[g] for g in m.gen_idxs if g == bus_idx])
                - current_load
                - sum(
                    [
                        m.Qij[br_idx[0], br_idx[1]]
                        for br_idx in m.branch_idxs
                        if br_idx[0] == bus_idx
                    ]
                )
                == 0.0
            )
        else:
            # return Constraint.Skip
            return (
                sum([m.Qg[g] for g in m.gen_idxs if g == bus_idx])
                - 0.0
                - sum(
                    [
                        m.Qij[br_idx[0], br_idx[1]]
                        for br_idx in m.branch_idxs
                        if br_idx[0] == bus_idx
                    ]
                )
                == 0.0
            )

    m.nodal_active = Constraint(m.bus_idxs, rule=nodal_active_rule)
    m.nodal_reactive = Constraint(m.bus_idxs, rule=nodal_reactive_rule)

    ## Line limits
    def line_limits_rule(m, i, j):
        return m.Pij[i, j] ** 2 + m.Qij[i, j] ** 2 <= S[i, j] ** 2

    m.branch_limits = Constraint(m.branch_idxs, rule=line_limits_rule)

    ## Phase angle difference: in a different way
    def phase_diff_rule(m, i, j):
        return abs(m.Va[i] - m.Va[j]) <= 2 * np.pi

    m.phase_diff = Constraint(m.branch_idxs, rule=phase_diff_rule)

    # Objective
    def objective_rule(m):
        groups_cost = net["poly_cost"].groupby(pd.Grouper("et")).groups
        costs_groups = []
        for group_key, group_idxs in groups_cost.items():
            curr_group = net["poly_cost"].loc[group_idxs]
            # active1   = curr_group['cp1_eur_per_mw'].values
            # active2   = curr_group['cp2_eur_per_mw'].values
            # reactive1 = curr_group['cq1_eur_per_mvar'].values
            # reactive2 = curr_group['cp2_eur_per_mw'].values
            curr_gen_idxs = (
                net[group_key].iloc[curr_group["element"].values]["bus"].values
            )
            # costs_groups.append([m.Pg[g] * 1 for g in curr_gen_idxs])
            for curr_element in curr_group["element"].values:
                gen_idx = net[group_key].iloc[curr_element]["bus"]
                cost_p0 = curr_group[curr_group["element"] == curr_element][
                    "cp0_eur"
                ].values[0]
                cost_p1 = curr_group[curr_group["element"] == curr_element][
                    "cp1_eur_per_mw"
                ].values[0]
                cost_p2 = curr_group[curr_group["element"] == curr_element][
                    "cp2_eur_per_mw2"
                ].values[0]
                cost_q0 = curr_group[curr_group["element"] == curr_element][
                    "cq0_eur"
                ].values[0]
                cost_q1 = curr_group[curr_group["element"] == curr_element][
                    "cq1_eur_per_mvar"
                ].values[0]
                cost_q2 = curr_group[curr_group["element"] == curr_element][
                    "cq2_eur_per_mvar2"
                ].values[0]
                # if gen_idxs == 0:
                #     costs_groups.append(
                #         cost_q0  # * m.UC[gen_idx]
                #         + cost_p0  # * m.UC[gen_idx]
                #         + m.Pg[gen_idx] * cost_p1
                #         + (m.Pg[gen_idx] ** 2) * cost_p2
                #         + m.Qg[gen_idx] * cost_q1
                #         + (m.Qg[gen_idx] ** 2) * cost_q2
                #     )
                # else:
                costs_groups.append(
                    cost_q0 * m.UC[gen_idx]
                    + cost_p0 * m.UC[gen_idx]
                    + m.Pg[gen_idx] * cost_p1
                    + (m.Pg[gen_idx] ** 2) * cost_p2
                    + m.Qg[gen_idx] * cost_q1
                    + (m.Qg[gen_idx] ** 2) * cost_q2
                )
                # costs_groups.append(m.Pg[gen_idx] * cost_p1 )
        cost = (
            sum(costs_groups)
            # + net["poly_cost"]["cp0_eur"].sum()
            # + net["poly_cost"]["cq0_eur"].sum()
        )
        return cost

    m.obj = Objective(rule=objective_rule)

    return m
