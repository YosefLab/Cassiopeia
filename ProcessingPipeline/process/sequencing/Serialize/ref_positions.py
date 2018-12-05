from Sequencing import utilities
from Circles import variants

extension = 'txt'

def _consolidate_counts(positions_list):
    positions_list = sorted(positions_list)
    consolidated_list = []
    for position, items in utilities.group_by(positions_list,
                                              key=lambda x: x[:4],
                                             ):
        ref_seq_name, ref_pos, ref_char, read_char = position
        count = sum(item[4] for item in items)
        consolidated = (ref_seq_name, ref_pos, ref_char, read_char, count)
        consolidated_list.append(consolidated)
    return consolidated_list

def write_file(positions_list, file_name):
    positions_list = _consolidate_counts(positions_list)
    with open(file_name, 'w') as fh:
        for ref_seq_name, ref_pos, ref_char, read_char, count in positions_list:
            line = '{0}\t{1}\t{2}\t{3}\t{4}\n'.format(ref_seq_name,
                                                      ref_pos,
                                                      ref_char,
                                                      read_char,
                                                      count,
                                                     )
            fh.write(line)

def read_file(file_name):
    def line_to_position(line):
        ref_seq_name, ref_pos, ref_char, read_char, count = line.strip().split('\t')
        ref_pos = int(ref_pos)
        count = int(count)
        return ref_seq_name, ref_pos, ref_char, read_char, count

    with open(file_name) as fh:
        positions_list = map(line_to_position, fh)

    return positions_list

def combine_data(first_positions_list, second_positions_list):
    combined_positions_list = first_positions_list + second_positions_list
    return combined_positions_list

def group_by_type(ref_positions_file_name, by_type_file_name):
    # Since write_file does _consolidate_counts, anything produced by read_file
    # will be consolidated.
    positions_list = read_file(ref_positions_file_name)

    type_lists = {change: [] for change in variants.change_order[:12]}

    for ref_seq_name, ref_pos, ref_char, read_char, count in positions_list:
        type_lists[ref_char, read_char].append((ref_seq_name, ref_pos, count))

    with open(by_type_file_name, 'w') as by_type_file:
        for ref_char, read_char in variants.change_order[:12]:
            by_type_file.write('{0} -> {1}\n'.format(ref_char, read_char))
            for ref_seq_name, ref_pos, count in type_lists[ref_char, read_char]:
                line = '\t{0}\t{1}\t{2}\n'.format(ref_seq_name,
                                                  ref_pos,
                                                  count,
                                                 )
                by_type_file.write(line)
