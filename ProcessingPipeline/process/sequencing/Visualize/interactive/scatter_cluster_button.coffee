models = cb_obj.document._all_models_by_name._dict

formatters =
    'clustered': {clustered_formatter}
    'original': {original_formatter}

cluster = not models['dendrogram'].visible
models['dendrogram'].visible = cluster

if cluster
    order_key = 'clustered'
else
    order_key = 'original'

data = models['heatmap_source'].data

for k in ['r', 'x', 'x_name', 'y', 'y_name', 'color']
    data[k] = data[k + '_' + order_key]

for axis in models['heatmap_axis']
    axis.formatter.code = formatters[order_key]

# Determine the new selection from the scatte axis labels.
x_name = models['x_axis'].axis_label
y_name = models['y_axis'].axis_label
num_pairs = data['x_name'].length
x_names = data['x_name']
y_names = data['y_name']
index = (i for i in [0..num_pairs] when x_names[i] == x_name and y_names[i] == y_name)[0]

models['heatmap_source'].selected.indices = [index]

models['heatmap_source'].change.emit()
models['heatmap_fig'].change.emit()
