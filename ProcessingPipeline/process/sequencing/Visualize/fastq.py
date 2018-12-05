import matplotlib.pyplot as plt
import numpy as np

from .. import fastq
from .. import utilities
from .. import adapters
from . import optional_ax, igv_colors, blues

@optional_ax
def plot_quality_histograms(quality_counts, ax=None):
    ''' Plots full distribution of quality values seens across all reads
        at each cycle.
    '''
    quality_counts = np.array(quality_counts, dtype=np.float)

    image = ax.imshow(quality_counts.T,
                      origin='lower',
                      interpolation='nearest',
                      cmap=blues,
                     )
    ax.set_xlabel('Cycle index')
    ax.set_ylabel('Quality score')

def plot_paired_quality_histograms(R1_quality_counts, R2_quality_counts):
    num_cycles, num_q_scores = R1_quality_counts.shape
    fig, (R1_ax, R2_ax) = plt.subplots(2, 1, figsize=(0.1 * num_cycles, 0.1 * (num_q_scores * 2 + 10)))
    plot_quality_histograms(R1_quality_counts, ax=R1_ax)
    plot_quality_histograms(R2_quality_counts, ax=R2_ax)
    R1_ax.set_title('R1')
    R2_ax.set_title('R2')

def plot_joint_average_quality_distribution(joint_average_q_distribution):
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(joint_average_q_distribution,
              origin='lower',
              interpolation='nearest',
              cmap=blues,
             )
    ax.set_xlabel('R2 average quality')
    ax.set_ylabel('R1 average quality')
    
def plot_data_statistics(R1_base_counts,
                         R2_base_counts,
                         R1_qualities,
                         R2_qualities,
                         R1_expected_seq=None,
                         R2_expected_seq=None,
                         R1_bracket_ranges=None,
                         R2_bracket_ranges=None,
                        ):
    ''' Plot fractions of all base calls that are each base at each cycle.
    '''
    fig, (R1_ax, R2_ax) = plt.subplots(2, 1, figsize=(20, 10))

    plot_composition(R1_base_counts, ax=R1_ax, expected_seq=R1_expected_seq, bracket_ranges=R1_bracket_ranges)
    plot_composition(R2_base_counts, ax=R2_ax, expected_seq=R2_expected_seq, bracket_ranges=R2_bracket_ranges)
    R1_ax.set_title('R1')
    R2_ax.set_title('R2')

    for ax in (R1_ax, R2_ax):
        ax.set_ylabel('Base composition')

    R1_q_ax = R1_ax.twinx()
    R2_q_ax = R2_ax.twinx()
    
    R1_mean_qs = [utilities.mean_from_histogram(row) for row in R1_qualities]
    R2_mean_qs = [utilities.mean_from_histogram(row) for row in R2_qualities]

    style = {'linestyle': '-',
             'color': 'black',
             'linewidth': 2,
             'alpha': 0.3,
            }

    R1_q_ax.plot(R1_mean_qs, **style)
    R2_q_ax.plot(R2_mean_qs, **style)

    for ax in [R1_q_ax, R2_q_ax]:
        ax.set_ylabel('Average quality score', rotation=90 + 180, va='bottom')
        ax.set_ylim(0, 41)
    
    total = R1_base_counts.sum(axis=1)[0]
    plt.suptitle('Base composition vs. cycle index\n{0:,d} total reads'.format(total))

@optional_ax
def plot_composition(base_counts,
                     bases_before=0,
                     expected_seq=None,
                     bracket_ranges=None,
                     ax=None,
                     save_as=None,
                    ):
    ''' Plot fractions of all base calls that are each base at each cycle.
    '''
    cycles, _ = base_counts.shape
    denominators = np.maximum(1, base_counts.sum(axis=1)).astype(float)
    total = int(denominators[0])

    base_order = utilities.base_order
    
    xs = np.arange(-bases_before, cycles - bases_before)
    
    for i, base in enumerate(base_order):
        fractions = base_counts[:, i] / denominators
        line_style = {'linestyle': '-',
                      'linewidth': 0.5,
                      'alpha': 0.5,
                      'color': igv_colors[base],
                     }
        marker_style = {'marker': '.',
                        'color': igv_colors[base],
                        'label': base,
                        'linestyle': 'None',
                       }
        ax.plot(xs, fractions, **marker_style) 
        if not expected_seq:
            ax.plot(xs, fractions, **line_style) 
    
    if expected_seq:
        start, sequence = expected_seq
        shade_background(start, sequence, ax=ax)

    if bracket_ranges:
        for text, (start, end) in bracket_ranges:
            # Truncate brackets that would extend off of the axis.
            if start > max(xs):
                continue
            end = min(end, max(xs) + 1)
            draw_range_bracket(ax, start, end - 1, text)
    
    ax.set_ylim(0, 1.01)
    ax.axhline(y=1, color='black', alpha=0.5)
    ax.set_xlim(min(xs) - 0.5, max(xs) + 0.5)

    ax.legend(loc='center left',
              bbox_to_anchor=(1.04, 0.5),
              framealpha=0.5,
              numpoints=1,
             )

@optional_ax
def shade_background(start, sequence, ax=None, save_as=None):
    ''' Lightly shade the background according to the expected sequence.
    '''
    for p, expected_bases in enumerate(sequence):
        expected_bases = [k for k, _ in utilities.group_by(expected_bases)]
        increment = 1. / len(expected_bases)
        for i, base in enumerate(expected_bases):
            ax.axvspan(start + p - 0.5,
                       start + p + 0.5,
                       ymax=1 - i * increment,
                       ymin=1 - (i + 1) * increment,
                       facecolor=igv_colors.normalized_rgbs[base],
                       alpha=0.3,
                       linewidth=0.7,
                      )

@optional_ax
def plot_adapter_composition(base_counts,
                             bases_before,
                             expected_seq,
                             adapter_ranges,
                             qualities,
                             ax=None,
                             save_as=None,
                            ):
    num_after_start = min(120, len(qualities) - bases_before)
    plot_composition(base_counts[:bases_before + num_after_start],
                     bases_before=bases_before,
                     ax=ax,
                     expected_seq=(0, expected_seq),
                     bracket_ranges=adapter_ranges,
                    )
    
    q_ax = ax.twinx()

    mean_qs = [utilities.mean_from_histogram(row) for row in qualities[:bases_before + num_after_start]]

    style = {'linestyle': '-',
             'color': 'black',
             'linewidth': 2,
             'alpha': 0.3,
            }
    q_ax.set_autoscale_on(False)
    xs = np.arange(-bases_before, num_after_start)
    q_ax.plot(xs, mean_qs, **style)
    q_ax.set_ylim(0, 41)

def plot_mean_position_qs(position_type_counts, xs, ax):
    position_average_qs = variants.compute_average_qualities(position_type_counts)
    _, max_qual_index, _, _ = position_type_counts.shape

    style = {'linestyle': '-',
             'color': 'black',
             'linewidth': 1,
             'alpha': 0.3,
            }
    
    ax.set_autoscale_on(False)
    ax.plot(xs, position_average_qs, **style)
    ax.set_ylim(0, max_qual_index - 1)
    ax.set_ylabel('Average quality score')
    
@optional_ax
def plot_average_qualities(average_q_distribution, label=None, save_as=None, ax=None):
    average_q_distribution = np.array(average_q_distribution, dtype=float)
    average_q_distribution /= average_q_distribution.sum()
    ax.plot(average_q_distribution, 'o-', label=label)
    ax.set_xlim(0, len(average_q_distribution) - 1)
    ax.set_xlabel('Average quality score')
    ax.set_ylabel('Fraction of reads')

@optional_ax
def plot_paired_average_qualities(R1_average_q_distribution, R2_average_q_distribution, name=None, save_as=None, ax=None):
    if name == None:
        R1_label = 'R1'
        R2_label = 'R2'
    else:
        R1_label = '{}_R1'.format(name)
        R2_label = '{}_R2'.format(name)
    plot_average_qualities(R1_average_q_distribution, label=R1_label, ax=ax)
    plot_average_qualities(R2_average_q_distribution, label=R2_label, ax=ax)
    ax.legend(loc='upper left', framealpha=0.5)

def remaining_Ns(base_counts, fig_file_name, paired=False):
    ''' Plot fractions of all base calls that are each base at each cycle.
    '''
    cycles, _ = base_counts.shape
    denominators = np.maximum(1, base_counts.sum(axis=1)).astype(float)
    total = int(denominators[0])

    fig = plt.figure()
    if paired:
        R1_counts = base_counts[:cycles / 2, :]
        R2_counts = base_counts[cycles / 2:, :]
        axs = [fig.add_subplot(2, 1, 1), fig.add_subplot(2, 1, 2)]
    else:
        raise NotImplementedError

    bases = fastq.base_order
    colors = 'mgbcrr'
    for ax, counts in zip(axs, [R1_counts, R2_counts]):
        counts_remaining = counts[::-1].cumsum(axis=0)[::-1]
        denoms = counts_remaining.sum(axis=1)
        fractions_remaining = np.true_divide(counts_remaining, denoms[:, np.newaxis])
        for i, (base, color) in enumerate(zip(bases, colors)):
            ax.plot(fractions_remaining[:, i], 'o-', color=color, linewidth=0.5, label=base) 
        ax.set_ylim(0, 1)
        ax.legend(loc='upper right', framealpha=0.5)
    
    plt.suptitle('Base composition vs. cycle index\n{0:,d} total reads'.format(total))
    fig.set_size_inches( (15, 15) )
    #plt.savefig(fig_file_name)
    #plt.close()
