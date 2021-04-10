import matplotlib.pyplot as plt


def plot_comparison(metrics):   
    fig, axs = plt.subplots(2, 2, figsize=(22, 16))

    # Population
    axs[0, 0].set_title('Population size')
    metrics.loc['acs']['population'].plot(label='ACS', ax=axs[0, 0])
    metrics.loc['acs2']['population'].plot(label='ACS2', ax=axs[0, 0])
    metrics.loc['acs2_ga']['population'].plot(label='ACS2_GA', ax=axs[0, 0])
    metrics.loc['yacs']['population'].plot(label='YACS', ax=axs[0, 0])
    metrics.loc['dynaq']['population'].plot(label='DynaQ', ax=axs[0, 0])
    axs[0, 0].legend(loc='best', frameon=False)

    # Knowledge
    axs[0, 1].set_title('Knowledge')
    metrics.loc['acs']['knowledge'].plot(label='ACS', ax=axs[0, 1])
    metrics.loc['acs2']['knowledge'].plot(label='ACS2', ax=axs[0, 1])
    metrics.loc['acs2_ga']['knowledge'].plot(label='ACS2_GA', ax=axs[0, 1])
    metrics.loc['yacs']['knowledge'].plot(label='YACS', ax=axs[0, 1])
    metrics.loc['dynaq']['knowledge'].plot(label='DynaQ', ax=axs[0, 1])
    axs[0, 1].legend(loc='lower right', frameon=False)

    # Generalization
    axs[1, 0].set_title('Generalization')
    metrics.loc['acs']['generalization'].plot(label='ACS', ax=axs[1, 0])
    metrics.loc['acs2']['generalization'].plot(label='ACS2', ax=axs[1, 0])
    metrics.loc['acs2_ga']['generalization'].plot(label='ACS2_GA', ax=axs[1, 0])
    metrics.loc['yacs']['generalization'].plot(label='YACS', ax=axs[1, 0])
    metrics.loc['dynaq']['generalization'].plot(label='DynaQ', ax=axs[1, 0])
    axs[1, 0].legend(loc='best', frameon=False)

    # Trial time
    metrics.groupby('agent')['time'].mean().plot.bar(ax=axs[1, 1])
    axs[1, 1].set_title('Average trial time')
    axs[1, 1].set_ylabel('Seconds [s]')
    axs[1, 1].set_xlabel('Agent')
