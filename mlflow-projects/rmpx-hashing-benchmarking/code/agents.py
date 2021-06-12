import lcs.agents.acs as acs
import lcs.agents.acs2 as acs2
import lcs.agents.yacs as yacs


def run_acs(env, trials, params):
    cfg = acs.Configuration(**params)
    agent = acs.ACS(cfg)
    return agent.explore_exploit(env, trials)


def run_acs2(env, trials, params):
    cfg = acs2.Configuration(**params)
    agent = acs2.ACS2(cfg)
    return agent.explore_exploit(env, trials)


def run_yacs(env, trials, params):
    cfg = yacs.Configuration(**params)
    agent = yacs.YACS(cfg)
    return agent.explore_exploit(env, trials)
