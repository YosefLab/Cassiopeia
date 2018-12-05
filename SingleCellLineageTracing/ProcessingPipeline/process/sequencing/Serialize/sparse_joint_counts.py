from collections import Counter

def _read_gene(fh, name_line):
    name, num_lines = name_line.strip().split()
    num_lines = int(num_lines)
    gene = Counter()
    for i in range(num_lines):
        key, count = fh.readline().strip().split()
        count = int(count)
        x, y = map(int, key.split(','))
        gene[x, y] = count

    return name, gene

def _write_gene(name, gene, fh):
    name_line = '{0}\t{1}\n'.format(name, len(gene))
    fh.write(name_line)
    for (x, y), count in gene.most_common():
        fh.write('{0},{1}\t{2}\n'.format(x, y, count))

def read_file(file_name):
    genes = {}
    fh = open(file_name)
    name_line = fh.readline()
    while name_line:
        name, gene = _read_gene(fh, name_line)
        genes[name] = gene
        name_line = fh.readline()

    return genes

def write_file(genes, file_name):
    with open(file_name, 'w') as fh:
        for name in sorted(genes):
            _write_gene(name, genes[name], fh)

def combine_data(first_genes, second_genes):
    for name in second_genes:
        if name not in first_genes:
            first_genes[name] = second_genes[name]
        else:
            first_genes[name].update(second_genes[name])

    return first_genes
