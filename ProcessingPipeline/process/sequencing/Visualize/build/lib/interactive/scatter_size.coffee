models = cb_obj.document._all_models_by_name._dict
models['scatter'].glyph.size = cb_obj.value
# Don't understand why, but only updates when scatter_source is triggered.
models['scatter_source'].change.emit()
