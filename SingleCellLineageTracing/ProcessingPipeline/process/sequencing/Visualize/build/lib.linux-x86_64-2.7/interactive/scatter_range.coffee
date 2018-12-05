models = cb_obj.document._all_models_by_name._dict

lower_bound = {lower_bound}
upper_bound = {upper_bound}

x_range = models['x_range']
y_range = models['y_range']

x_range.start = lower_bound if x_range.start < lower_bound
x_range.end = upper_bound if x_range.end > upper_bound

y_range.start = lower_bound if y_range.start < lower_bound
y_range.end = upper_bound if y_range.end > upper_bound
