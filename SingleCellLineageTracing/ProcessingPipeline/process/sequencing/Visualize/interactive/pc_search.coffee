models = cb_obj.document._all_models_by_name._dict
models['clear_constraints'].callback.func(models['clear_constraints'], null, require, exports)

search_string = cb_obj.value

data = models['labels'][0].source.data
dqs = (name.indexOf(search_string) == -1 for name in data['label'])

for label in models['labels']
    label.source.data['dq'] = dqs

models['hover_tool'].callback.func(models['hover_tool'], 'force_redraw', require, exports)
