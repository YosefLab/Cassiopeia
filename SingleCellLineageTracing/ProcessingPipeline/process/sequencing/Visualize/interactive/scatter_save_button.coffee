models = cb_obj.document._all_models_by_name._dict

column_names = {column_names}

filtered_data = models['filtered_source'].data

# learned from http://stackoverflow.com/questions/14964035/how-to-export-javascript-array-info-to-csv-on-client-side
csv_content = "data:text/csv;charset=utf-8,"

lines = [column_names.join('\t')]

first_name = column_names[0]
length = filtered_data[first_name].length

for i in [0...length]
    line = (filtered_data[name][i] for name in column_names).join('\t')
    lines.push line

csv_content += lines.join('\n')

encoded = encodeURI(csv_content)
link = document.createElement('a')
link.setAttribute('href', encoded)
link.setAttribute('download', 'table_contents.txt')
link.click()