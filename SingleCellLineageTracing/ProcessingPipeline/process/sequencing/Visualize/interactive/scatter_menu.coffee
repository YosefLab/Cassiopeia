models = cb_obj.document._all_models_by_name._dict

scatter_data = models['scatter_source'].data
label_data = models['filtered_source'].data
hist_data = models['histogram_source'].data

squeeze = (possibly_array) ->
    if Array.isArray(possibly_array)
        squeezed = possibly_array[0]
    else
        squeezed = possibly_array
    return squeezed

for axis in ['x', 'y']
    name = squeeze models[axis + '_menu'].value

    scatter_data[axis] = scatter_data[name]
    label_data[axis] = label_data[name]

    for suffix in ['_all', '_bins_left', '_bins_right']
        hist_data[axis + suffix] = hist_data[name + suffix]

    models[axis + '_axis'].axis_label = name

# Call to recompute selection histograms.
models['scatter_selection_callback'].func(models['scatter_source'], cb_data, require, exports)

models['scatter_source'].change.emit()
models['filtered_source'].change.emit()
models['histogram_source'].change.emit()
