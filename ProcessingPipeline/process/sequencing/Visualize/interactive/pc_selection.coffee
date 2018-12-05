models = cb_obj.document._all_models_by_name._dict
by_y_dqs = (source_by_y.data['dq'] for name, source_by_y of models when name.startsWith('source_by_y'))

x = cb_obj.name['source_by_x_'.length...]

indices = cb_obj.selected.indices
if indices.length > 0
    by_x_dq = (true for val in cb_obj.data['dq'])
    by_x_dq[i] = false for i in indices

    cb_obj.data['dq'] = by_x_dq
    for y, dq of by_x_dq
        by_y_dqs[y][x] = dq

    # Don't show a bar if all points in the column are in the selection.
    if indices.length == cb_obj.data['dq'].length
        bottom = 0
        top = 0
    else
        vals = (cb_obj.data['y'][i] for i in indices)
        bottom = Math.min(vals...) - 10
        top = Math.max(vals...) + 10
    
    x_int = parseInt(x)

    vbar_data = models['constraints'].data
    index = vbar_data['x'].indexOf(x_int)
    if index == -1
        index = vbar_data['x'].length

    vbar_data['x'][index] = x_int
    vbar_data['top'][index] = top
    vbar_data['bottom'][index] = bottom
    vbar_data['width'][index] = 0.2

cb_obj.selected.indices = []

models['search'].value = ''
