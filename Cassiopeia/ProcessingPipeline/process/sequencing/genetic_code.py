import Bio.Data.CodonTable
import copy

nucleotide_order = 'TCAG'
nucleotide_to_index = {b: i for i, b in enumerate(nucleotide_order)}
def codon_sorting_key(codon):
    return [nucleotide_to_index[b] for b in codon]

forward_table = Bio.Data.CodonTable.standard_dna_table.forward_table

non_stop_codons = sorted(forward_table, key=codon_sorting_key)
stop_codons = sorted(Bio.Data.CodonTable.standard_dna_table.stop_codons, key=codon_sorting_key)

full_forward_table = copy.copy(forward_table)
for stop_codon in stop_codons:
    full_forward_table[stop_codon] = '*'

all_codons = sorted(non_stop_codons + stop_codons, key=codon_sorting_key)
codon_to_index = {codon: i for i, codon in enumerate(all_codons)}

full_back_table = {}
for codon in all_codons:
    amino_acid = full_forward_table[codon]
    if amino_acid not in full_back_table:
        full_back_table[amino_acid] = [codon]
    else:
        full_back_table[amino_acid].append(codon)

amino_acids = sorted(full_back_table)

codon_to_amino_acid_and_index = {}
for amino_acid in full_back_table:
    for i, codon in enumerate(full_back_table[amino_acid]):
        codon_to_amino_acid_and_index[codon] = (amino_acid, i)

degeneracy = {amino_acid: len(full_back_table[amino_acid]) for amino_acid in amino_acids}

def codons_from_seq(seq):
    for i in range(0, len(seq), 3):
        yield seq[i:i + 3]

anticodon_to_codons = {
    'IGC': {'GCU', 'GCC'},
    'UGC': {'GCA', 'GCG'},
    'ICG': {'CGU', 'CGC', 'CGA'},
    'CCG': {'CGG'},
    'UCU': {'AGA'},
    'CCU': {'AGG'},
    'GUU': {'AAU', 'AAC'},
    'GUC': {'GAU', 'GAC'},
    'GCA': {'UGU', 'UGC'},
    'UUG': {'CAA'},
    'CUG': {'CAG'},
    'UUC': {'GAA'},
    'CUC': {'GAG'},
    'GCC': {'GGU', 'GGC'},
    'UCC': {'GGA'},
    'CCC': {'GGG'},
    'GUG': {'CAU', 'CAC'},
    'IAU': {'AUU', 'AUC'},
    'UAU': {'AUA'},
    'UAA': {'UUA'},
    'CAA': {'UUG'},
    'GAG': {'CUU', 'CUC'},
    'UAG': {'CUA', 'CUG'},
    'UUU': {'AAA'},
    'CUU': {'AAG'},
    'CAU': {'AUG'},
    'GAA': {'UUU', 'UUC'},
    'IGG': {'CCU', 'CCC'},
    'UGG': {'CCA', 'CCG'},
    'IGA': {'UCU', 'UCC'},
    'UGA': {'UCA'},
    'CGA': {'UCG'},
    'GCU': {'AGU', 'AGC'},
    'IGU': {'ACU', 'ACC'},
    'UGU': {'ACA'},
    'CGU': {'ACG'},
    'CCA': {'UGG'},
    'GUA': {'UAU', 'UAC'},
    'IAC': {'GUU', 'GUC'},
    'UAC': {'GUA'},
    'CAC': {'GUG'},
}

u_to_t = str.maketrans('U', 'T')
def rna_to_dna(rna):
    return rna.translate(u_to_t)

t_to_u = str.maketrans('T', 'U')
def dna_to_rna(dna):
    return dna.translate(t_to_u)

codon_to_anticodon = {}

for anticodon in anticodon_to_codons:
    dna_anticodon = rna_to_dna(anticodon)
    for codon in anticodon_to_codons[anticodon]:
        dna_codon = rna_to_dna(codon)
        codon_to_anticodon[dna_codon] = dna_anticodon
