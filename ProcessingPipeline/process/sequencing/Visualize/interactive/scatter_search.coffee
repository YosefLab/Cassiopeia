models = cb_obj.document._all_models_by_name._dict

column_names = {column_names}

case_sensitive = models['case_sensitive'].active.length > 0
if not case_sensitive
    query = cb_obj.value.toLowerCase()
    possibly_lowercase = (t) -> t.toLowerCase()
else
    query = cb_obj.value
    possibly_lowercase = (t) -> t

all_matches = []
if query != ''
    for column in column_names
        targets = models['scatter_source'].data[column]
        matches = (i for t, i in targets when possibly_lowercase(t).indexOf(query) > -1 and i not in all_matches)
        all_matches.push matches...

models['scatter_source'].selected.indices = all_matches
models['scatter_selection_callback'].func(models['scatter_source'], 'from_search', require, exports)
