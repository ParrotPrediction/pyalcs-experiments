from pylab import cm
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

# Common color palette
palette = cm.get_cmap('Paired', 5)

COLORS = {
    'acs': palette(0),
    'acs2': palette(1),
    'acs2_ga': palette(2),
    'yacs': palette(3),
    'dynaq': palette(4),
}


def plot_comparison(metrics):   
    fig, axs = plt.subplots(2, 2, figsize=(22, 16))
    
    metrics['knowledge_100'] = metrics['knowledge'] * 100
    metrics['generalization_100'] = metrics['generalization'] * 100

    # Population
    metrics.loc['acs']['population'].plot(label='ACS', color=COLORS['acs'], ax=axs[0, 0])
    metrics.loc['acs2']['population'].plot(label='ACS2', color=COLORS['acs2'], ax=axs[0, 0])
    metrics.loc['acs2_ga']['population'].plot(label='ACS2_GA', color=COLORS['acs2_ga'], ax=axs[0, 0])
    metrics.loc['yacs']['population'].plot(label='YACS', color=COLORS['yacs'], ax=axs[0, 0])
    metrics.loc['dynaq']['population'].plot(label='DynaQ', color=COLORS['dynaq'], ax=axs[0, 0])
    axs[0, 0].set_title('Population size')
    axs[0, 0].set_ylabel('Number of rules/classifiers')
#     axs[0, 0].legend(loc='best', frameon=False)

    # Knowledge
    axs[0, 1].set_title('Knowledge')
    metrics.loc['acs']['knowledge_100'].plot(label='ACS', color=COLORS['acs'], ax=axs[0, 1])
    metrics.loc['acs2']['knowledge_100'].plot(label='ACS2', color=COLORS['acs2'], ax=axs[0, 1])
    metrics.loc['acs2_ga']['knowledge_100'].plot(label='ACS2_GA', color=COLORS['acs2_ga'], ax=axs[0, 1])
    metrics.loc['yacs']['knowledge_100'].plot(label='YACS', color=COLORS['yacs'], ax=axs[0, 1])
    metrics.loc['dynaq']['knowledge_100'].plot(label='DynaQ', color=COLORS['dynaq'], ax=axs[0, 1])
#     axs[0, 1].legend(loc='lower right', frameon=False)
    axs[0, 1].yaxis.set_major_formatter(mtick.PercentFormatter())

    # Generalization
    axs[1, 0].set_title('Generalization')
    metrics.loc['acs']['generalization_100'].plot(label='ACS', color=COLORS['acs'], ax=axs[1, 0])
    metrics.loc['acs2']['generalization_100'].plot(label='ACS2', color=COLORS['acs2'], ax=axs[1, 0])
    metrics.loc['acs2_ga']['generalization_100'].plot(label='ACS2_GA', color=COLORS['acs2_ga'], ax=axs[1, 0])
    metrics.loc['yacs']['generalization_100'].plot(label='YACS', color=COLORS['yacs'], ax=axs[1, 0])
    metrics.loc['dynaq']['generalization_100'].plot(label='DynaQ', color=COLORS['dynaq'], ax=axs[1, 0])
#     axs[1, 0].legend(loc='best', frameon=False)
    axs[1, 0].yaxis.set_major_formatter(mtick.PercentFormatter())

    # Trial time
    times = metrics.groupby('agent')['time'].mean().to_dict()
    
    labels = ['ACS', 'ACS2', 'ACS2_GA', 'YACS', 'DynaQ']
    values = [times['acs'], times['acs2'], times['acs2_ga'], times['yacs'], times['dynaq']]
    colors = [COLORS['acs'], COLORS['acs2'], COLORS['acs2_ga'], COLORS['yacs'], COLORS['dynaq']]
    
    axs[1, 1].bar(labels, values, color=colors)
    axs[1, 1].set_title('Average trial time')
    axs[1, 1].set_ylabel('Seconds [s]')
    
    # Global legend
    handles, labels = axs[1, 0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=len(values), loc='lower center')
