#!/usr/bin/env perl

use strict;
use warnings;

#die "\nuse: $0 bamFile outfile.txt\n\n" if $#ARGV!=1;

# Note: Stripped seqs need to be fixed depending on direction of intBC

my $samtool = "samtools";
my $bamfile = $ARGV[0];
my $outfile = $ARGV[1];

open(OFILE,">$outfile");

print OFILE "cellBC\tUMI\tintBC\treadCount\tcigar\tAS\tNM\tr1\tr2\tr3\ttargetSite\treadName\tr1_no_context\tr2_no_context\tr3_no_context\n";

my $totalReads = 0;
my $noBCReads = 0;
my $totalUMIs = 0;
my $noBCUMIs = 0;
my $corrected = 0;
my $seenBadBCs = 0;


open(IN,"$samtool view -F 4 $bamfile |") or die($!);
while(<IN>) {
	chomp;
	my ($qname,$flag,$rname,$pos,$mapq,$cigar,$rnext,$pnext,$tlen,$seq,$qual,$ASStr,$NMStr, $URStr, $ARStr, $COStr) = split(/\t/,$_);
	my ($cellBC,$UMI,$readCount) = split(/_/,$qname);

	my ($t,$i,$AS) = split(/:/,$ASStr); # alignment score
	($t,$i,my $NM) = split(/:/,$NMStr); # alignment distance to reference
	(my $at, $i, my $AR) = split(/:/, $ARStr); # Allele record string (treated as genomic locus for UMI ec)
	(my $ut, $i, my $UR) = split(/:/, $URStr); # Error corrected UMI

	# if UR and AR got switched, try switching them back
	if ($ut ne "UR") { 
		($ut, $at) = ($at, $ut);
		($UR, $AR) = ($AR, $UR);
		# print STDERR "$ut\t$at\t$UR\t$AR\n";
	}

	die "\nIncorrect UR Tag\n\n" if $ut ne "UR";
	my @COSplit = split(/:/, $COStr); # Information containing all allele infomration
	my $CO = join(":", @COSplit[2..$#COSplit]);

	my ($intBC, $r1Cigar, $r2Cigar, $r3Cigar, $r1Cigar_old, $r2Cigar_old, $r3Cigar_old) = split(/;/, $CO);
	#print STDERR "COStr: $COStr\n";
	#print STDERR "URStr: $URStr\n";
	#print STDERR "ARStr: $ARStr\n";
	#print STDERR "$CO\n";
	#print STDERR "$r1Cigar, $r2Cigar, $r3Cigar, $r1Cigar_old, $r2Cigar_old, $r3Cigar_old\n";

	print OFILE "$cellBC\t$UR\t$intBC\t$readCount\t$cigar\t$AS\t$NM\t$r1Cigar\t$r2Cigar\t$r3Cigar\t$seq\t$qname\t$r1Cigar_no_context\t$r2Cigar_no_context\t$r3Cigar_no_context\n";
	$totalReads+=$readCount;
	$totalUMIs+=1;
	if ($intBC eq "NC") {
		$noBCReads+=$readCount;
		$noBCUMIs+=1;
	}
}
close IN;
close OFILE;

my $propNoBCs = $noBCReads/$totalReads*100;
my $propNoBCsUMI = $noBCUMIs/$totalUMIs*100;
my $propCorrected = 0;
if  ($seenBadBCs > 0) {
	$propCorrected = $corrected/($seenBadBCs)*100;
}
print "totalReads: $totalReads noBCReads: $noBCReads (" . sprintf("%.1f",$propNoBCs) . "%) ";
print "totalUMIs: $totalUMIs noBCUMIs: $noBCUMIs (" . sprintf("%.1f",$propNoBCsUMI) . "%)\n";
print "Corrected intBCs: $corrected of $seenBadBCs (" . sprintf("%.1f",$propCorrected) . "%)\n";
