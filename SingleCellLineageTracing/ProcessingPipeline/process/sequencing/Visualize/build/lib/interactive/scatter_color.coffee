models = cb_obj.document._all_models_by_name._dict

data = models['scatter_source'].data
choice = cb_obj.value

if choice == ''
    main_key = '_black'
    selection_key = '_orange'
else
    main_key = choice
    selection_key = choice

data['_color'] = data[main_key]
data['_selection_color'] = data[selection_key]

models['scatter_source'].change.emit()
