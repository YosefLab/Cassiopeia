#!/usr/bin/env perl

use strict;
use warnings;

die "\nuse: $0 bamFile outfile.txt\n\n" if $#ARGV!=1;

my $samtool = "samtools";
my $bamfile = $ARGV[0];
my $outfile = $ARGV[1];

open(OFILE,">$outfile");

#open(IN,"$samtool view $bamfile |") or die($!);
open(IN,"$samtool view -f 4 $bamfile |") or die($!); # only get unmapped reads, modified 08/2017
#my $counter=0;
while(<IN>) {
	#last if $counter==100;
	chomp;
	my ($qname,$flag,$rname,$pos,$mapq,$cigar,$rnext,$pnext,$tlen,$seq,$qual,@flags) = split(/\t/,$_);
	my %flagDict;
	foreach my $f (@flags) {
		my ($tag, $type, $val) = split(/:/,$f);
		$flagDict{$tag}=$val;
	}
	if (exists $flagDict{"CB"} && exists $flagDict{"UB"}) {
		my $readName = $qname . "_" . $flagDict{"CB"} . "_" . $flagDict{"UB"};
		print OFILE "\@$readName\n$seq\n+\n$qual\n";
	}
	#$counter++;
}
close IN;
close OFILE;
