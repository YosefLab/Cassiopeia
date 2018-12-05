models = cb_obj.document._all_models_by_name._dict

# cb_data is used to flag if this callback was triggered manually by
# the search button or subset menu callback's. If not, reset the values of those
# widgets.

# To prevent this from erasing a selection that was just made, store indices
# and re-assign them afterwards.
indices = cb_obj.selected.indices

if cb_data == 'from_heatmap'
else
    if (models['search']?) and cb_data != 'from_search'
        models['search'].value = ''
    if (models['subset_menu']?) and cb_data != 'from_subset'
        models['subset_menu'].value = ''

cb_obj.selected.indices = indices

# Make the histograms of all data slightly darker if nothing is selected. 
if indices.length == 0
    models['hist_x_all'].glyph.fill_alpha = 0.2
    models['hist_y_all'].glyph.fill_alpha = 0.2
else
    models['hist_x_all'].glyph.fill_alpha = 0.1
    models['hist_y_all'].glyph.fill_alpha = 0.1

full_data = models['scatter_source'].data
filtered_data = models['filtered_source'].data

for key, values of full_data
    filtered_data[key] = (values[i] for i in indices)

models['filtered_source'].change.emit()#('change')

if (models['table']?)
    models['table'].change.emit()#('change')

get_domain_info = (name) ->
    bins_left = models['histogram_source'].data[name + '_bins_left']
    bins_right = models['histogram_source'].data[name + '_bins_right']
    bounds = [bins_left[0], bins_right[bins_right.length - 1]]
    domain_info =
        bins: bins_left
        bounds: bounds
    return domain_info

binned_to_counts = (binned) -> (b.length for b in binned)

loaded =
    'd3': if d3? then d3 else null

update_bins = () ->
    for name in ['x', 'y']
        domain_info = get_domain_info(name)
        # histogram behavior is a little unintuitive.
        # Through trial and error, appears that domain needs to be given min
        # and max data value, and thresholds needs to be given array that
        # include min but not max
        binner = loaded['d3'].histogram().domain(domain_info.bounds).thresholds(domain_info.bins)
        data = filtered_data[name]
        binned = binner(data)
        counts = binned_to_counts(binned)
        models['histogram_source'].data[name + '_selected'] = counts
    
    models['histogram_source'].change.emit()

if d3? and d3.histogram
    update_bins()
    return

if not (window.requirejs?)
    return

`
requirejs.config({{
    paths: {{
        d3: "https://d3js.org/d3-array.v1.min"
    }}
}});

requirejs(['d3'], function(d3) {{
    loaded['d3'] = d3;
    update_bins();
}});
`
