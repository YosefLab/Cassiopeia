#!/usr/bin/env perl

use strict;
use warnings;

die "\nuse: $0 MSACollapsed.txt 3'PrimerSeq\n\n" if $#ARGV!=1;

my $str3 = $ARGV[1];
#my $str3 = "GCTTCGTACGCGAAACTAGCGT"; # PCT31-v3-tS3lib

open(AFILE,$ARGV[0]);
my $header=<AFILE>;
while(<AFILE>) {
	chomp;
	my ($cellBC,$UMI,$readCount,$seq) = split(/\t/);
	$seq=~s/\s+$//g;
	print ">${cellBC}_${UMI}_$readCount\n${seq}$str3\n";
}
close AFILE;


