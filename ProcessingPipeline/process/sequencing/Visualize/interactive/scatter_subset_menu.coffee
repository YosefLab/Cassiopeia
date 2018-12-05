models = cb_obj.document._all_models_by_name._dict

subset_indices = {subset_indices}

query = cb_obj.value

selection = []
if query != ''
    selection = subset_indices[query]

models['scatter_source'].selected.indices = selection
models['scatter_selection_callback'].func(models['scatter_source'], 'from_subset', require, exports)
