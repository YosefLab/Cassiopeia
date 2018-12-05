from itertools import cycle

import numpy as np
import pandas as pd
import bokeh.plotting
import bokeh.io
import bokeh.models

from .external_coffeescript import build_callback

color_list = bokeh.palettes.Category20[20]

def pc(df):
    fig = bokeh.plotting.figure(width=1500,
                                height=1200,
                                tools=['reset', 'undo', 'save', 'pan', 'wheel_zoom', 'tap'],
                               )

    box_select = bokeh.models.tools.BoxSelectTool(name='box_select')
    box_select.callback = build_callback('pc_box_select')
    fig.add_tools(box_select)
    fig.toolbar.active_drag = box_select

    if df.columns.nlevels == 2:
        names = ['{0}: {1}'.format(gn, en) for gn, en in df.columns.values]
    else:
        names = df.columns

    lines = []
    for y, ((row_name, row), color) in enumerate(zip(df.iterrows(), cycle(color_list))):
        vals = row.values
        data = {
            'x': list(range(len(vals))),
            'y': vals,
            'name': [row_name]*len(names),
            'dq': [False for _ in vals],
        }

        source = bokeh.models.ColumnDataSource(data=data, name='source_by_y_{0}'.format(y))

        line = fig.line(x='x', y='y',
                        source=source,
                        line_alpha=0.7,
                        line_width=3,
                        color=color,
                        name='line_{0}'.format(y),
                       )
        line.hover_glyph = line.glyph
        lines.append(line)

    columns = df.columns.values

    colors = [c for c, _ in zip(cycle(color_list), df.index)]

    for x, exp_name in enumerate(columns):
        vals = df[exp_name].values
        data = {
            'x': [x for _ in vals],
            'y': vals,
            'dq': [False for _ in vals],
            'color': colors,
            'size': [6 for _ in vals],
            'alpha': [0.9 for _ in vals],
        }

        source = bokeh.models.ColumnDataSource(data=data, name='source_by_x_{0}'.format(x))
        circles = fig.circle(x='x', y='y',
                             source=source,
                             color='color',
                             fill_alpha='alpha',
                             line_alpha=0,
                             size='size',
                            )
        source.callback = build_callback('pc_selection')
        
    for x in [0, len(columns) - 1]:
        vals = df.iloc[:, x].values

        if x == 0:
            text_align = 'right'
            x_offset = -10
        else:
            text_align = 'left'
            x_offset = 10
            
        data = {
            'x': [x for _ in vals],
            'y': vals,
            'label': df.index.values,
            'hover_color': colors,
            'text_color': ['black' for _ in vals],
            'text_alpha': [0.9 for _ in vals],
            'highlight': [False for _ in vals],
            'dq': [False for _ in vals],
        }

        source = bokeh.models.ColumnDataSource(data=data)

        labels = bokeh.models.LabelSet(x='x',
                                       y='y',
                                       text='label',
                                       level='glyph',
                                       x_offset=x_offset,
                                       y_offset=0,
                                       source=source,
                                       text_font_size='12pt',
                                       text_align=text_align,
                                       text_alpha='text_alpha',
                                       text_color='text_color',
                                       text_baseline='middle',
                                       name='labels',
                                       text_font='monospace',
                                      )
        fig.add_layout(labels)
        
    hover = bokeh.models.HoverTool(line_policy='interp',
                                   renderers=lines,
                                   tooltips=None,
                                   name='hover_tool',
                                  )
    hover.callback = build_callback('pc_hover')
    fig.add_tools(hover)
    fig.xgrid.visible = False

    fig.xaxis.minor_tick_line_color = None

    code = '''
    dict = {dict}
    return dict[tick]
    '''.format(dict=dict(enumerate(names)))

    fig.xaxis.ticker = bokeh.models.FixedTicker(ticks=list(range(len(names))))
    fig.xaxis.formatter = bokeh.models.FuncTickFormatter(code=code)
    fig.xaxis.major_label_text_font_size = '12pt'
    fig.xaxis.major_label_orientation = np.pi / 4

    fig.yaxis.axis_label = 'nts between cut'
    fig.yaxis.axis_label_text_font_style = 'normal'
    fig.yaxis.axis_label_text_font_size = '16pt'
    fig.yaxis.major_label_text_font_size = '12pt'

    fig.x_range = bokeh.models.Range1d(-0.1 * len(names), 1.1 * (len(names) - 1))
    fig.x_range.callback = bokeh.models.CustomJS.from_coffeescript(code=f'''
    if cb_obj.start < {-0.1 * len(names)}
        cb_obj.start = {-0.1 * len(names)}

    if cb_obj.end > {1.1 * (len(names) - 1)}
        cb_obj.end = {1.1 * (len(names) - 1)}
    ''')

    fig.y_range.callback = bokeh.models.CustomJS.from_coffeescript(code=f'''
    if cb_obj.start < 0
        cb_obj.start = 0
        
    if cb_obj.end > {df.max().max() * 1.05}
        cb_obj.end = {df.max().max() * 1.05} 
    ''')

    clear_constraints = bokeh.models.widgets.Button(label='Clear constraints',
                                                    width=50,
                                                    name='clear_constraints',
                                                   )
    clear_constraints.callback = build_callback('pc_clear')

    data = dict(x=[], width=[], top=[], bottom=[])
    constraint_source = bokeh.models.ColumnDataSource(data=data, name='constraints')
    vbar = bokeh.models.glyphs.VBar(x='x',
                                    top='top',
                                    bottom='bottom',
                                    width='width',
                                    fill_alpha=0.15,
                                    fill_color='black',
                                    line_width=0,
                                    line_alpha=0,
                                   )

    fig.add_glyph(constraint_source, vbar, selection_glyph=vbar, nonselection_glyph=vbar)

    fig.min_border = 100

    text_input = bokeh.models.widgets.TextInput(title='Highlight:', name='search')
    text_input.callback = build_callback('pc_search')

    columns = [
        bokeh.layouts.column([fig]),
        bokeh.layouts.column([bokeh.layouts.Spacer(height=100), clear_constraints, text_input]),
    ]

    bokeh.io.show(bokeh.layouts.row(columns))
