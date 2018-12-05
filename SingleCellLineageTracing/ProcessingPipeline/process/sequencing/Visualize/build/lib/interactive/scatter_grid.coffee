models = cb_obj.document._all_models_by_name._dict

choice = cb_obj.labels[cb_obj.active]

for grid in models['grid']
    grid.visible = choice == 'grid'

diagonals = models['diagonal']
if not Array.isArray(diagonals)
    diagonals = [diagonals]

for diagonal in diagonals
    diagonal.visible = choice == 'diagonal'
