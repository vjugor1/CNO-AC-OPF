import pandapower.networks as nw

grid_name_map = {}
#!!! nw is somehow invisible, so I make it thourgh converter
grid_name_map["case4"] = nw.case4gs
grid_name_map["case6"] = nw.case6ww
grid_name_map["case9"] = nw.case9
grid_name_map["case14"] = nw.case14
grid_name_map["case24"] = nw.case24_ieee_rts
grid_name_map["case30"] = nw.case_ieee30
grid_name_map["case39"] = nw.case39
grid_name_map["case57"] = nw.case57
grid_name_map["case89"] = nw.case89pegase
grid_name_map["case118"] = nw.case118
grid_name_map["case118i"] = nw.iceland
grid_name_map["case145"] = nw.case145
grid_name_map["case200"] = nw.case_illinois200
grid_name_map["case300"] = nw.case300
grid_name_map["case1354"] = nw.case1354pegase
grid_name_map["case1888"] = nw.case1888rte
grid_name_map["case2224"] = nw.GBnetwork
grid_name_map["case2848"] = nw.case2848rte
grid_name_map["case2869"] = nw.case2869pegase
grid_name_map["case3120"] = nw.case3120sp
grid_name_map["case6470"] = nw.case6470rte
grid_name_map["case6495"] = nw.case6495rte
grid_name_map["case6515"] = nw.case6515rte
grid_name_map["case9241"] = nw.case9241pegase
