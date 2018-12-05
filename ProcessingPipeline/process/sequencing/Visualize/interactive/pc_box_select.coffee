models = cb_obj.document._all_models_by_name._dict
lines = (v for k, v of models when k.startsWith('line'))

ys = []
for line in lines
    y = line.name['line_'.length...]

    highlight = not line.data_source.data['dq'].some((x) -> x)
    if highlight
        ys.push(y)
    
dqs = (true for line in lines)
dqs[y] = false for y in ys

for labels in models['labels']
    labels.source.data['dq'] = dqs

models['constraints'].change.emit()
models['hover_tool'].callback.func(models['hover_tool'], 'force_redraw', require, exports)
