models = cb_obj.document._all_models_by_name._dict
lines = (v for k, v of models when k.startsWith('line'))

for name, line_source of models when name.startsWith('source_by_y')
    line_source.data['dq'] = (false for c in line_source.data['dq'])

for name, circle_source of models when k.startsWith('source_by_x_')
    circle_source.data['dq'] = (false for l in lines)

for label in models['labels']
    label.source.data['dq'] = (false for l in lines)
    
models['constraints'].data = {{'x': [], 'width': [], 'top': [], 'bottom': []}}
models['constraints'].change.emit()

models['hover_tool'].callback.func(models['hover_tool'], 'force_redraw', require, exports)
