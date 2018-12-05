models = cb_obj.document._all_models_by_name._dict

data = models['labels'][0].source.data

if cb_data == 'force_redraw'
    change = true
    highlight = false
else
    y = parseInt cb_data.renderer.name['line_'.length...]

    highlight = cb_data.index.line_indices.length > 0
    change = data['highlight'][y] != highlight

if change
    data['highlight'][y] = highlight
    dqs = data['dq']

    line_widths = ((if dq then 2 else 3) for dq in dqs)
    circle_sizes = ((if dq then 4 else 6) for dq in dqs)
    colors = ('black' for dq in dqs)

    if highlight
        colors[y] = data['hover_color'][y]

        line_alphas = ((if dq then 0.1 else 0.3) for dq in dqs)
        line_alphas[y] = 0.9

        text_alphas = ((if dq then 0.01 else 0.05) for dq in dqs)
        text_alphas[y] = 0.95

        line_widths[y] = 6
        circle_sizes[y] = 8

    else
        line_alphas = ((if dq then 0.1 else 0.7) for dq in dqs)
        text_alphas = ((if dq then 0.05 else 0.9) for dq in dqs)

    for label in models['labels']
        label.source.data['text_alpha'] = text_alphas
        label.source.data['text_color'] = colors
        label.source.change.emit()

    circle_sources = (v for k, v of models when k.startsWith('source_by_x_'))
    for circle_source in circle_sources
        circle_source.data['size'] = circle_sizes
        circle_source.data['alpha'] = line_alphas
        circle_source.change.emit()

    for alpha, y in line_alphas
        models['line_' + y].glyph.line_alpha = alpha
        models['line_' + y].glyph.line_width = line_widths[y]
        models['line_' + y].change.emit()
