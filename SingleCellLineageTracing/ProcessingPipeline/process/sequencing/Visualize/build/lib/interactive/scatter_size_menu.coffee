models = cb_obj.document._all_models_by_name._dict

choice = cb_obj.value

if choice == ''
    choice = '_uniform_size'

models['scatter_source'].data['_size'] = models['scatter_source'].data[choice]
models['scatter_source'].change.emit()
