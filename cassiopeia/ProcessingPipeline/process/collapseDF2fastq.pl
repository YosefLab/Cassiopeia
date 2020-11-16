#!/usr/bin/env perl

use strict;
use warnings;

die "\nuse: $0 txtFile outfile.fastq\n\n" if $#ARGV!=1;

my $txtfile = $ARGV[0];
my $outfile = $ARGV[1];

open(OFILE,">$outfile");

open(my $fh, '<', $txtfile) or die($!);
my $first = <$fh>;

while(my $row = <$fh>) {
	chomp $row;
	my ($index, $cellBC,$UMI,$readCount,$grpFlag,$seq,$qual,$readName,$filter) = split(/\t/,$row);
	print OFILE "\@$cellBC\_$UMI\_$readCount\n$seq\n+\n$qual\n";
}
close $fh;
close OFILE;
