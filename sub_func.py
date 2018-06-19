def name2num(agent_name):
    return int(agent_name[6:8])

def num2name(agent_num):
    return "Agent[" + "{0:02d}".format(agent_num) + "]"

    