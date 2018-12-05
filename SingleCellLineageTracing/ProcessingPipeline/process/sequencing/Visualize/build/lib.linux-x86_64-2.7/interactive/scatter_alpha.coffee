models = cb_obj.document._all_models_by_name._dict

value = cb_obj.value
if typeof value == 'string'
    value = parseFloat(value)

models['scatter'].glyph.fill_alpha = value

models['scatter_source'].change.emit()
