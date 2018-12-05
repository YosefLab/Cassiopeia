import os
import glob
from collections import defaultdict

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image
import scipy.signal
import scipy.interpolate
import numpy as np
import ipywidgets
import yaml

from sequencing import utilities

def load_image(image_fn):
    return matplotlib.image.imread(image_fn)

def load_labels(image_fn):
    labels_fn = '{0}_labels.txt'.format(image_fn[:-len('.tif')])
    labels = [line.strip() for line in open(labels_fn)]
    return labels

def load_annotations(image_fn):
    root, ext = os.path.splitext(image_fn)
    annotation_fn = '{0}.yaml'.format(root)

    if os.path.exists(annotation_fn):
        with open(annotation_fn) as annotation_fh:
            annotations = yaml.load(annotation_fh)
    else:
        annotations = {}

    return annotations

def save_annotations(image_fn, annotations):
    root, ext = os.path.splitext(image_fn)
    annotation_fn = '{0}.yaml'.format(root)

    with open(annotation_fn, 'w') as annotation_fh:
        yaml.dump(annotations, annotation_fh, default_flow_style=False)

def identify_lane_boundaries(image, num_lanes=None):
    ''' Finds lane boundaries by summing image columns, identifying peaks in
    window-summed column sums, then determining widths around peaks at which
    signal has returned to background.
    '''
    cols = image.sum(axis=0)
    window_sums = sum_over_window(cols, 20)
    peaks, = scipy.signal.argrelmax(window_sums, order=16)

    if num_lanes is not None:
        descending = sorted(peaks, key=lambda p: window_sums[p], reverse=True)
        peaks = sorted(descending[:num_lanes])

    mins_between = find_mins_between(cols, peaks)
    half_widths = []
    for p, (l, r) in zip(peaks, mins_between):
        right_threshold = cols[r] + 0.05 * (cols[p] - cols[r])
        # argmax gives index of first true value
        right_width = np.argmax(cols[p:r] <= right_threshold)
        
        left_threshold = cols[l] + 0.05 * (cols[p] - cols[l])
        left_width = np.argmax(cols[l:p + 1][::-1] <= left_threshold)
    
        half_width = max(right_width, left_width)
        half_widths.append(half_width)

    half_width = np.mean(half_widths) 
    either_side = int(half_width)
    boundaries = [(p - either_side, p + either_side) for p in peaks]
    return boundaries

def extract_profiles(image, boundaries):
    profiles = []
    for start, end in boundaries:
        start = max(start, 0)
        lane = image[:, start:end + 1]
        rows, cols = lane.shape
        raw = lane.sum(axis=1)[::-1]
        normalized = np.true_divide(raw, cols)
        profiles.append(normalized)
    return profiles

def extract_background(image, boundaries):
    profiles = []
    for (_, end), (start, _) in zip(boundaries[:-1], boundaries[1:]):
        lane = image[:, end + 1 + 1:start + 1 - 1]
        rows, cols = lane.shape
        raw = lane.sum(axis=1)[::-1]
        normalized = np.true_divide(raw, cols)
        profiles.append(normalized)
    return profiles

def sum_over_window(array, either_side):
    summed = np.zeros(len(array))
    cumulative = np.concatenate(([0], np.cumsum(array)))
    window = 2 * either_side + 1
    summed[either_side:-either_side] = cumulative[window:] - cumulative[:-window]
    return summed

def get_ladder100_peaks(ys):
    ys = np.asarray(ys)
    lengths = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1200, 1517]
    
    relative_maxes, = scipy.signal.argrelmax(ys, order=4)

    # Filter out peaks that aren't sufficiently high.
    median = np.percentile(ys, 50)
    overall_max = np.max(ys)
    threshold = median + (overall_max - median) * 0.3
    high_enough = relative_maxes[ys[relative_maxes] > threshold] 

    # Filter out peaks that are too close to the top or bottom.
    candidates = [c for c in high_enough if 0.05 * len(ys) < c < 0.95 * len(ys)]

    if len(candidates) == len(lengths):
        peaks = dict(zip(lengths, candidates))
    else:
        # Peaks are probably too close together to call.
        # Give up on the higher ones.
        peaks = dict(zip(lengths[:5], candidates[:5]))
        peaks[lengths[-1]] = candidates[-1]
    
        relative_maxes,  = scipy.signal.argrelmax(ys, order=10)
        between = [x for x in relative_maxes if peaks[500] < x < peaks[1517]]
        if len(between) > 0:
            peaks[1000] = max(between, key=lambda x: ys[x])
        
    return peaks

def find_mins_between(ys, peaks):
    ''' For each index in peaks, returns the indices of the minimum values
    between peaks on either side.
    '''
    either_side = []
    for i, p in enumerate(peaks):
        if i == 0:
            left = 0
        else:
            left = peaks[i - 1]

        if i == len(peaks) - 1:
            right = len(ys)
        else:
            right = peaks[i + 1]

        min_left = left + np.argmin(ys[left:p])
        min_right = p + np.argmin(ys[p:right])
        either_side.append((min_left, min_right))

    return either_side

def compute_peak_to_troughs(ys, peaks):
    ''' For each index in peaks, returns the peak-to-trough ratio of
    the value at that peak to the average of the minimum values between
    the peak and its adjacent peaks.
    '''
    mins_between = find_mins_between(ys, peaks)
    trough_means = [np.mean([ys[left], ys[right]]) for left, right in mins_between]
    ratios = [ys[p] / trough_mean for p, trough_mean in zip(peaks, trough_means)]
    return ratios

def get_ladder10_peaks(ys, ladder100_peaks):
    # Strategy: identify the 100 bp beak by finding the relative max closest
    # to the 100 bp peak in the 100bp ladder.
    # March up and down relative maxes from there to assign 10 bp peaks.
    # To avoid false positive peaks between the lower ladder components,
    # require them to be the max in a larger index winow (via the order
    # argument to argrelmax).
    
    ys = np.asarray(ys)
    peaks = {}
    
    def get_relative_max_xs(order):
        xs, = scipy.signal.argrelmax(ys, order=order)
        return xs

    big_peak_xs = get_relative_max_xs(50)
    peaks[100] = min(big_peak_xs, key=lambda x: abs(ladder100_peaks[100] - x))
    
    # below 100
    xs = get_relative_max_xs(10)
    peak_to_troughs = compute_peak_to_troughs(ys, xs)
    below_100 = [x for x, r in zip(xs, peak_to_troughs)
                 if x < peaks[100] and r > 1.1
                ]
    
    for length, x in zip(np.arange(90, 10, -10), below_100[::-1]):
        peaks[length] = x
    
    # above 100
    xs = get_relative_max_xs(3)
    above_100 = [x for x in xs if x > peaks[100]]
    
    for length, x in zip(np.arange(110, 160, 10), above_100):
        peaks[length] = x
        
    return peaks

def make_conversion_functions(lanes):
    ladder100_peaks = get_ladder100_peaks(lanes['ladder100'])
    ladder10_peaks = get_ladder10_peaks(lanes['ladder10'], ladder100_peaks)
    for l in [100, 110, 120, 130, 140, 150]:
        ladder10_peaks.pop(l, None)
    length_to_x = dict(ladder10_peaks.items() + ladder100_peaks.items())
    lengths = sorted(length_to_x)
    xs = [length_to_x[l] for l in lengths]

    log_length_to_x = scipy.interpolate.interp1d(np.log10(lengths), xs, 'linear', fill_value='extrapolate')
    length_to_x = lambda l: log_length_to_x(np.log10(l))
    
    x_to_log_length = scipy.interpolate.interp1d(xs, np.log10(lengths), 'linear', fill_value='extrapolate')
    x_to_length = lambda x: 10**x_to_log_length(x)

    return length_to_x, x_to_length

def analyze_image(image, annotations):
    def contain_overlap(boundaries):
        overlapping = False
        for (_, end), (start, _) in utilities.pairwise(boundaries):
            if end >= start:
                overlapping = True
        return overlapping

    def invalid(boundaries, labels):
        wrong_number = labels is not None and len(labels) != len(boundaries)
        return wrong_number or contain_overlap(boundaries)

    labels = annotations.get('labels')
    if labels is None:
        num_lanes = None
    else:
        num_lanes = len(labels)

    boundaries = identify_lane_boundaries(image, num_lanes=num_lanes)
    profiles = extract_profiles(image, boundaries)
    
    if labels is None:
        labels = range(1, len(boundaries) + 1)
    labels = [str(l) for l in labels]

    #if invalid(boundaries, labels):
    #    labels = map(str, range(1, len(boundaries) + 1))

    lanes = dict(zip(labels, profiles))

    return labels, boundaries, profiles, lanes

def plot_gel(image_fn,
             vertical_range=(0, 1),
             highlight=None,
             contrast=1.0,
             show_expected='both',
             invert=False,
            ):
    image = load_image(image_fn)
    annotations = load_annotations(image_fn)

    labels, boundaries, profiles, lanes = analyze_image(image, annotations)

    fig_height = 8.
    plot_width = 12.
    rows, cols = image.shape
    im_width = fig_height * cols / rows

    gridspec_kw = dict(width_ratios=[im_width, plot_width], wspace=0)
    
    fig, (im_ax, line_ax) = plt.subplots(1, 2,
                                         figsize=(plot_width + im_width, fig_height),
                                         gridspec_kw=gridspec_kw,
                                        )

    kwargs = {
        'line': {
            'highlight': dict(alpha=0.95, linewidth=1),
            'nonhighlight': dict(alpha=0.1, linewidth=0.5),
            'uniform': dict(alpha=0.9, linewidth=0.7),
        },
        'boundary': {
            'highlight': dict(alpha=0.95, linewidth=1.2),
            'nonhighlight': dict(alpha=0.2, linewidth=1),
            'uniform': dict(alpha=0.5, linewidth=1),
        },
        'text': {
            'highlight': dict(alpha=0.95),
            'nonhighlight': dict(alpha=0.1),
            'uniform': dict(alpha=0.9),
        },
    }
    
    label_to_kwargs = defaultdict(dict)

    if highlight is None:
        highlight = []
    
    for i, label in enumerate(labels):
        color = 'C{0}'.format(i % 10)
        
        if len(highlight) > 0:
            if label in highlight:
                key = 'highlight'
            else:
                key = 'nonhighlight'
        else:
            key = 'uniform'
        
        for kind in kwargs:
            copy = kwargs[kind][key].copy()
            copy['color'] = color
            label_to_kwargs[kind][label] = copy
    
    for i, label in enumerate(labels):
        if 'ladder' in label and not label in highlight:
            continue

        ys = lanes[label]
        xs = np.arange(len(ys))

        line_ax.plot(xs, ys, 'o-', label=label, markersize=0.3, **label_to_kwargs['line'][label])

    if 'ladder100' in lanes:
        ladder100_peaks = get_ladder100_peaks(lanes['ladder100'])
        ladder10_peaks = get_ladder10_peaks(lanes['ladder10'], ladder100_peaks)
    
        # Only include the 100 bp peak from the 10 bp ladder.
        ladder100_peaks.pop(100)

        peaks = list(ladder100_peaks.items()) + list(ladder10_peaks.items())

        major_peaks = [
            100,
            200,
            500,
            1000,
            1517,
        ]

        for length, x in peaks:
            alpha = 0.3 if length in major_peaks else 0.05
            line_ax.axvline(x, color='black', alpha=alpha)

        line_ax.set_xticks([x for length, x in peaks])
        line_ax.set_xticklabels([str(length) for length, x in peaks], rotation=-90, ha='center', size=8)

        if show_expected:
            length_to_x, x_to_length = make_conversion_functions(lanes)
            expected = annotations.get('expected', {})
            for length, name in expected.items():
                x = length_to_x(length)
                if show_expected in {'both', 'plot'}:
                    line_ax.axvline(x, color='black', linestyle='--')
                    line_ax.annotate(name,
                                     xy=(x, 1),
                                     xycoords=('data', 'axes fraction'),
                                     xytext=(0, 2),
                                     textcoords='offset points',
                                     va='bottom',
                                     ha='center',
                                    )

                if invert:
                    line_color = 'black'
                else:
                    line_color = 'white'
                y = rows - x

                if show_expected in {'both', 'image'}:
                    im_ax.plot([0.99 * cols, cols], [y, y], color=line_color)
                    im_ax.plot([0, 0.01 * cols], [y, y], color=line_color)

                    im_ax.annotate(name,
                                   xy=(1, y),
                                   xycoords=('axes fraction', 'data'),
                                   xytext=(2, 0),
                                   textcoords='offset points',
                                   fontsize=8,
                                   va='center',
                                   ha='left',
                                  )

    else:
        line_ax.set_xticks([])

    line_ax.set_yticks([])
    
    
    if invert:
        im_cmap = matplotlib.cm.binary
    else:
        im_cmap = matplotlib.cm.binary_r

    image_range = image.max() - image.min()
    im_ax.imshow(image,
                 cmap=im_cmap,
                 vmin=image.min(),
                 vmax=image.min() + image_range * contrast,
                )
    im_ax.set_xticks([])
    im_ax.set_yticks([])
    
    for i, (label, (start, end)) in enumerate(zip(labels, boundaries)):
        im_ax.axvline(start, **label_to_kwargs['boundary'][label])
        im_ax.axvline(end, **label_to_kwargs['boundary'][label])
        im_ax.annotate(label,
                       xy=(np.mean((start, end)), 1),
                       xytext=(0, 2),
                       xycoords=('data', 'axes fraction'),
                       textcoords='offset points',
                       rotation=45,
                       fontsize=10,
                       va='bottom',
                       **label_to_kwargs['text'][label])
        
    #legend = line_ax.legend(loc='upper left', framealpha=0.5)
    #for text in legend.get_texts():
    #    label = text.get_text()
    #    text.set(**label_to_kwargs['text'][label])
   
    x_min, x_max = map(int, np.asarray(vertical_range) * len(xs))
    line_ax.set_xlim(x_min, x_max)
    
    im_ax.autoscale(False)

    if vertical_range != (0, 1):
        y_min, y_max = vertical_range

        if y_min == 0:
            y_min = 0.002
        if y_max == 1:
            y_max= 0.999

        if invert:
            line_color = 'black'
        else:
            line_color = 'white'

        line_kwargs = dict(transform=im_ax.transAxes, color=line_color)
        im_ax.plot([0.005, 0.005, 0.05], [y_min + 0.05, y_min, y_min], **line_kwargs)
        im_ax.plot([0.005, 0.005, 0.05], [y_max - 0.05, y_max, y_max], **line_kwargs)
        im_ax.plot([1 - 0.005, 1 - 0.005, 1 - 0.05], [y_min + 0.05, y_min, y_min], **line_kwargs)
        im_ax.plot([1 - 0.005, 1 - 0.005, 1 - 0.05], [y_max - 0.05, y_max, y_max], **line_kwargs)
        
    head, tail = os.path.split(image_fn)
    line_ax.set_title(tail, y=1.05)
    
    return fig

def plot_gels_interactive(image_fns, **kwargs):

    def make_tab(image_fn):
        def generate_figure(highlight, vertical_range, contrast, invert, show_expected):
            fig = plot_gel(image_fn,
                           highlight=highlight,
                           vertical_range=vertical_range,
                           contrast=contrast,
                           invert=invert,
                           show_expected=show_expected,
                           **kwargs)
            plt.show()
            return fig

        annotations = load_annotations(image_fn)
        labels = annotations.get('labels', [])

        widgets = {
            'vertical_range': ipywidgets.FloatRangeSlider(
                value=[0, 1],
                continuous_update=False,
                min=0,
                max=1,
                step=0.01,
                layout=ipywidgets.Layout(height='200px'),
                description='Vertical range:',
                style={'description_width': 'initial'},
                orientation='vertical',
            ),
            'highlight': ipywidgets.SelectMultiple(
                options=labels,
                value=[],
                layout=ipywidgets.Layout(height='200px'),
                description='Highlight:',
            ),
            'contrast': ipywidgets.FloatSlider(
                value=1.0,
                continuous_update=False,
                min=0,
                max=1,
                step=0.05,
                layout=ipywidgets.Layout(height='200px'),
                description='Contrast:',
                style={'description_width': 'initial'},
                orientation='vertical',
            ),
            'invert': ipywidgets.ToggleButton(
                value=False,
                description='Invert',
                icon='check',
            ),
            'show_expected': ipywidgets.Dropdown(
                options={'neither': False,
                         'plot': 'plot',
                         'image': 'image',
                         'both': 'both',
                        },
                value=False,
                description='Show expected',
            ),
            'save': ipywidgets.Button(
                description='Save snapshot',
            ),
            'file_name': ipywidgets.Text(
                value=os.environ['HOME'] + '/name.png',
            ),
            'close': ipywidgets.Button(
                description='Close tab',
            ),
            'notes': ipywidgets.Textarea(
                description='Notes:',
                layout=ipywidgets.Layout(height='200px', width='400px'),
            ),
            'labels': ipywidgets.Textarea(
                description='Lane labels:',
                layout=ipywidgets.Layout(height='200px', width='400px'),
            ),
            'expected': ipywidgets.Textarea(
                description='Expected:',
                layout=ipywidgets.Layout(height='200px', width='400px'),
            ),
            'update': ipywidgets.Button(
                description='Update',
            ),
        }

        def save(_):
            fig = interactive.result
            fn = widgets['file_name'].value
            fig.savefig(fn, bbox_inches='tight')

        widgets['save'].on_click(save)
        
        def close(_):
            titles = [tabs.get_title(i) for i, c in enumerate(tabs.children) if i != tabs.selected_index]
            tabs.children = [c for i, c in enumerate(tabs.children) if i != tabs.selected_index]
            for i, title in enumerate(titles):
                tabs.set_title(i, title)

        widgets['close'].on_click(close)

        interactive = ipywidgets.interactive(generate_figure, **widgets)
        output = interactive.children[-1]
        interactive.update()

        widgets['invert'].value = annotations.get('invert', False)
        widgets['contrast'].value = annotations.get('contrast', 1.0)
        widgets['notes'].value = annotations.get('notes', '')
        widgets['labels'].value = '\n'.join(annotations.get('labels', []))
        expected = annotations.get('expected', {})
        lines = ['{0}: {1}'.format(length, name) for length, name in sorted(expected.items())]
        widgets['expected'].value = '\n'.join(lines)

        def update_annotations(_):
            def get_lines(key):
                lines = map(str, widgets[key].value.split('\n'))
                if lines == ['']:
                    lines = []
                return lines

            annotations['notes'] = str(widgets['notes'].value)

            labels = get_lines('labels')
            widgets['highlight'].options = labels

            expected = {}
            for line in get_lines('expected'):
                length, name = line.split(': ')
                length = int(length)
                expected[length] = name

            annotations['expected'] = expected
            annotations['labels'] = labels
            annotations['invert'] = widgets['invert'].value
            annotations['contrast'] = widgets['contrast'].value

            save_annotations(image_fn, annotations)
            interactive.update()

        widgets['update'].on_click(update_annotations)

        def group_widgets(keys, kind):
            if kind == 'row':
                Box = ipywidgets.HBox
            elif kind == 'col':
                Box = ipywidgets.VBox

            keys = [widgets.get(k, k) for k in keys]

            return Box(keys)

        make_row = lambda keys: group_widgets(keys, 'row')
        make_col = lambda keys: group_widgets(keys, 'col')

        layout = make_col([
            make_row([output]),
            make_row(['highlight',
                      'vertical_range',
                      'contrast',
                      make_col([
                          'invert',
                      ]),
                      make_col([
                          'save',
                          'close',
                          'update',
                      ]),
                      make_col([
                          'file_name',
                          'show_expected',
                      ]),
                     ]),
            make_row(['labels',
                      'expected',
                      'notes',
                     ]),
        ])

        return layout 

    shorten_fn = lambda fn: os.path.split(fn)[1][:-len('_cropped.tif')]

    tabs = ipywidgets.Tab()
    tabs.children = [make_tab(image_fn) for image_fn in image_fns]
    for i, image_fn in enumerate(image_fns):
        shortened = shorten_fn(image_fn)
        tabs.set_title(i, shortened)
            
    fns = sorted(glob.glob('/home/jah/cropped/*.tif'))
    file_names = ipywidgets.Select(
        options=[(shorten_fn(fn), fn) for fn in fns],
        value=None,
        layout=ipywidgets.Layout(width='300px', height='800px'),
    )

    def open_file(change):
        existing_titles = [tabs.get_title(i) for i in range(len(tabs.children))]
        image_fn = change['new']
        shortened = shorten_fn(image_fn)
        if shortened not in existing_titles:
            new_tab = make_tab(image_fn)
            tabs.children = tabs.children + (new_tab,)
            tabs.set_title(len(tabs.children) - 1, shortened)
        
        # Make the selected file the active tab
        titles = [tabs.get_title(i) for i in range(len(tabs.children))]
        tabs.selected_index = titles.index(shortened)

    def sync_menu_selection(change):
        if change['new'] is None:
            # All tabs are closed.
            file_names.value = None
        else:
            shorteneds = [s for s, f in file_names.options]
            fulls = [f for s, f in file_names.options]
            new_title = tabs.get_title(change['new'])
            index = shorteneds.index(new_title)
            file_names.value = fulls[index]

    tabs.observe(sync_menu_selection, names='selected_index')

    file_names.observe(open_file, names='value')

    file_col = ipywidgets.VBox([file_names])

    return ipywidgets.HBox([file_col, tabs])
