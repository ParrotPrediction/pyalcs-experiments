import pandas as pd
from lcs import Perception
from matplotlib import pyplot as plt


def print_cl(cl):
    action = None
    if cl.action == 0:
        action = 'L'
    if cl.action == 1:
        action = 'R'
    return (
        f"{cl.condition} - {action} - {cl.effect}"
        f"[fit: {cl.fitness:.3f}, r: {cl.r:.2f}, ir: {cl.ir:.2f}]")


def calculate_knowledge(population, environment):
    transitions = environment.env.get_transitions()
    reliable = [c for c in population if c.is_reliable()]
    nr_correct = 0

    for start, action, end in transitions:
        p0 = Perception((str(start),))
        p1 = Perception((str(end),))

        if any([True for cl in reliable if
                cl.predicts_successfully(p0, action, p1)]):
            nr_correct += 1

    return nr_correct / len(transitions) * 100.0

def plot_performance(df, population):
    # https://github.com/rougier/matplotlib-cheatsheet/blob/master/README.md
    explore_exploit_trial = df.query('phase=="explore"').tail(1).index[0]

    # performance plots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 6),
                                                 sharex=True, sharey=False)
    # fig.tight_layout()

    # Steps
    avg_exploit_steps = df.query('phase=="exploit"')['steps_in_trial'].mean()

    ax1.set_title('Number of steps')
    ax1.set_ylabel('Steps')
    ax1.axvline(x=explore_exploit_trial, c='black', linestyle='dashed',
                linewidth=1)
    ax1.axhline(y=avg_exploit_steps, c='red', linestyle='dashed', linewidth=1,
                label=f'avg in exploit ({avg_exploit_steps} steps)')
    df['steps_in_trial'].plot(ax=ax1)
    ax1.legend()

    # Population
    ax3.set_title('Population size')
    ax3.set_ylabel('Count')
    ax3.axvline(x=explore_exploit_trial, c='black', linestyle='dashed',
                linewidth=1)
    df['population'].plot(ax=ax3, label='Macro-classifiers')
    df['reliable'].plot(ax=ax3, label='Reliable')
    ax3.legend()

    # Top classifiers
    for ax in [ax2, ax4]:
        ax.remove()

    axbig = fig.add_subplot(1, 3, 3)
    top = 15
    axbig.set_title(f'Top {top} classifiers (of {len(population)})')
    axbig.axis('off')
    for i, cl in enumerate(
            sorted(population, key=lambda cl: -cl.fitness)[:top]):
        pos_y = 0.9
        axbig.text(.01, pos_y - i * 0.05, print_cl(cl), fontsize=13)

