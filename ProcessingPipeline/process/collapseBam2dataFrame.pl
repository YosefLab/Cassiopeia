#!/usr/bin/env perl

use strict;
use warnings;

die "\nuse: $0 bamFile outfile.txt\n\n" if $#ARGV!=1;

my $samtool = "samtools";
my $bamfile = $ARGV[0];
my $outfile = $ARGV[1];

open(OFILE,">$outfile");
print OFILE "cellBC\tUMI\treadCount\tgrpFlag\tseq\tqual\treadName\n";

open(IN,"$samtool view $bamfile |") or die($!);

while(<IN>) {
	chomp;
	my ($readName,$flag,$rname,$pos,$mapq,$cigar,$rnext,$pnext,$tlen,$seq,$qual,@flags) = split(/\t/,$_);
	my ($cellBC,$UMI,$readCount,$grpFlag) = split(/_/,$readName);
	print OFILE "$cellBC\t$UMI\t$readCount\t$grpFlag\t$seq\t$qual\t$readName\n";
}
close IN;
close OFILE;