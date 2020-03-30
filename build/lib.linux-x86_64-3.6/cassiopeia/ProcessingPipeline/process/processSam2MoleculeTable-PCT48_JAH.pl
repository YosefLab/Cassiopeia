#!/usr/bin/env perl

use strict;
use warnings;

die "\nuse: $0 bamFile ref.fa outfile.txt\n\n" if $#ARGV!=2;

# Note: Stripped seqs need to be fixed depending on direction of intBC

my $samtool = "samtools";
my $bamfile = $ARGV[0];
my $reffile = $ARGV[1]; # reference alignment for intBC calling
my $outfile = $ARGV[2];

my @intBCLim = (21,34);
my $intBCLength = 14;

my @r1Lim = (95,118);  # ade2
my @r2Lim = (149,172); # bri1
my @r3Lim = (203,226); # whtB

my $ref = "";
open(REF, $reffile) or die($!);
while(<REF>) {
	chomp;
	my $s = substr($_, 0, 1);
	if ($s ne ">") {
		$ref = $_;
		last;
	}
}
close REF;
my $reflength = length $ref;
my $refanchor = substr($ref, 9, 10);
print "refseq: $ref\n";

open(OFILE,">$outfile");
print OFILE "cellBC\tUMI\tintBC\treadCount\tcigar\tAS\tNM\tr1\tr2\tr3\ttargetSite\treadName\tr1.old\tr2.old\tr3.old\n";

my $totalReads = 0;
my $noBCReads = 0;
my $totalUMIs = 0;
my $noBCUMIs = 0;
my $corrected = 0;
my $seenBadBCs = 0;

open(IN,"$samtool view -F 4 $bamfile |") or die($!);
while(<IN>) {
	chomp;
	my ($qname,$flag,$rname,$pos,$mapq,$cigar,$rnext,$pnext,$tlen,$seq,$qual,$ASStr,$NMStr) = split(/\t/,$_);
	my ($cellBC,$UMI,$readCount) = split(/_/,$qname);
	my $start = $pos;
	my @cigarChunks = $cigar =~ m/\d+[MIDNSHP=X]/g;
	my ($t,$i,$AS) = split(/:/,$ASStr); # alignment score
	($t,$i,my $NM) = split(/:/,$NMStr); # alignment distance to reference

	my $refItr=$start;
	my $queryItr=0;
	my $queryPad = 0;
	my $printedFlag = 0;

	my $r1Cigar="";
	my $r2Cigar="";
	my $r3Cigar="";
	my $r1Cigar_old = "";
	my $r2Cigar_old = "";
	my $r3Cigar_old = "";
	my $intBC="NC";
	my $strippedSeq = $seq;

	foreach my $c (@cigarChunks) {
		# match => increment iterator
		if ($c =~ m/(\d+)M/) {
			my $matchLen = $1; # length of match stretch
			my $refItrN = $refItr+$matchLen; # the match is in the interval between refItr and refItrN
			#print "refItr $refItr $refItrN barcode $intBCLim[0] $intBCLim[1]\n";

			# we're checking if the match goes into the barcode region
			# misses instances where entire bc isn't matched, ie. there is a 1bp deletion in the BC
			if ($intBCLim[0]>=$refItr && $intBCLim[1]<=$refItrN) {
				# integration BC within seq
				my $intBCOffset = $intBCLim[0]-$refItr;
				if ($queryPad>(-1 * $intBCLength)) { # entire BC is deleted so reported as "not captured"
					$intBC = substr($seq,$queryItr+$intBCOffset,$intBCLength+$queryPad);
					my $intBCEndIndex = $queryItr+$intBCOffset+$intBCLength+$queryPad;
					$strippedSeq = substr($seq,$intBCEndIndex,length($seq)-$intBCEndIndex);
				}
				#print STDERR "$intBCLim[0] refItr:$refItr queryItr:$queryItr matchLen:$matchLen intBCOffset:$intBCOffset \n";
			}
			# next part is a patch but I think it could be fixed using query pad
			elsif ($refItr>=$intBCLim[0] && $refItr<=$intBCLim[1] && $refItrN>$intBCLim[1]) { # partial 5' deletion of intBC
				my $newIntBCLen = $intBCLim[1]-$refItr+1;
				$intBC = substr($seq,$queryItr,$newIntBCLen);
				my $intBCEndIndex = $queryItr+$newIntBCLen;
				$strippedSeq = substr($seq,$intBCEndIndex,length($seq)-$intBCEndIndex);
			}
			elsif ($refItrN>=$intBCLim[0] && $refItrN<=$intBCLim[1] && $refItr<$intBCLim[0]) { # partial 3' deletion of intBC
				my $newIntBCLen = $refItrN-$intBCLim[0]+1;
				my $intBCOffset = $intBCLim[0]-$refItr;
				$intBC = substr($seq,$queryItr+$intBCOffset,$newIntBCLen);
				my $intBCEndIndex = $queryItr+$intBCOffset+$newIntBCLen;
				$strippedSeq = substr($seq,$intBCEndIndex,length($seq)-$intBCEndIndex);
			}
			$refItr+=$matchLen;
			$queryItr+=$matchLen;
		} elsif ($c =~ m/(\d+)I/) {
		# insertion in read => add to insertions, increment itr
			my $size=$1;
			$queryPad = $size if $refItr==$intBCLim[0];
			my $start = $queryItr;
			$queryItr+=$size;

			# get context of insertion
			my $context_l = substr($seq, $start-7, 7);
			my $context_r = substr($seq, $start, 7 + $size);

			# insertions must lie within one region
			if ($refItr>=$r1Lim[0] && $refItr<=$r1Lim[1]) {
				$r1Cigar_old .= "${refItr}:${size}I";
				$r1Cigar.="${context_l}[${refItr}:${size}I]${context_r}";
			}
			elsif ($refItr>=$r2Lim[0] && $refItr<=$r2Lim[1]) {
				$r2Cigar.="${context_l}[${refItr}:${size}I]${context_r}";
				$r2Cigar_old .="${refItr}:${size}I";
			}
			elsif ($refItr>=$r3Lim[0] && $refItr<=$r3Lim[1]) {
				$r3Cigar.="${context_l}[${refItr}:${size}I]${context_r}";
				$r3Cigar_old .="${refItr}:${size}I";
			}
		}
		# deletion in read => add to dels, increment refItr
		elsif ($c =~ m/(\d+)D/) {
			my $size=$1;
			my $refItrLast=$refItr;
			$refItr+=$size;
			$queryPad = -1 * $size if $refItr==$intBCLim[0]; #deletion starts at intBC

			# get context of deletion
			my $context_l = substr($seq, $queryItr-7, 7);
			my $context_r = substr($seq, $queryItr, 7);

			# deletions can span multiple regions
			if (($r1Lim[0]<=$refItr && $refItr<=$r1Lim[1]) ||
			   ($r1Lim[0]<=$refItrLast && $refItrLast<=$r1Lim[1]) ||
			   ($refItrLast<=$r1Lim[0] && $refItr>=$r1Lim[1]) ||
			   ($refItrLast>=$r1Lim[0] && $refItr<=$r1Lim[1])) {
				$r1Cigar.="${context_l}[${refItrLast}:${size}D]${context_r}";
				$r1Cigar_old .="${refItrLast}:${size}D";
			}

			if (($r2Lim[0]<=$refItr && $refItr<=$r2Lim[1]) ||
			   ($r2Lim[0]<=$refItrLast && $refItrLast<=$r2Lim[1]) ||
			   ($refItrLast<=$r2Lim[0] && $refItr>=$r2Lim[1]) ||
			   ($refItrLast>=$r2Lim[0] && $refItr<=$r2Lim[1])) {
				$r2Cigar.="${context_l}[${refItrLast}:${size}D]${context_r}";
				$r2Cigar_old .="${refItrLast}:${size}D";

			}

			if (($r3Lim[0]<=$refItr && $refItr<=$r3Lim[1]) ||
			   ($r3Lim[0]<=$refItrLast && $refItrLast<=$r3Lim[1]) ||
			   ($refItrLast<=$r3Lim[0] && $refItr>=$r3Lim[1]) ||
			   ($refItrLast>=$r3Lim[0] && $refItr<=$r3Lim[1])) {
				$r3Cigar.="${context_l}[${refItrLast}:${size}D]${context_r}";
				$r3Cigar_old .="${refItrLast}:${size}D";

			}
		}
		elsif ($c =~ m/(\d+)H/) { # hard clipping, does not appear in seq
			# hard clipped, do nothing for now
			my $size = $1;
			print STDERR "$qname $cigar $pos $seq $size\n";
		}
		elsif ($c =~ m/\d+[NSP=X]/) {
			print STDERR "Unknown CIGAR Occurance: $c $qname\n";
		}
	}
	if ($intBC eq "NC" || (length $intBC) < $intBCLength) {
		my $anchor = substr($seq, 9, 10);

		if ($anchor eq $refanchor) {
			$intBC = substr($seq, 20, $intBCLength);
			print "Corrected intBC: $intBC\n";
			$corrected++;
		}
		$seenBadBCs++;
	}

	$r1Cigar="None" if $r1Cigar eq "";
	$r2Cigar="None" if $r2Cigar eq "";
	$r3Cigar="None" if $r3Cigar eq "";
	$r1Cigar_old = "None" if $r1Cigar_old eq "";
	$r2Cigar_old = "None" if $r2Cigar_old eq "";
	$r3Cigar_old = "None" if $r3Cigar_old eq "";

	#print OFILE "$cellBC\t$UMI\t$intBC\t$readCount\t$r1Cigar\t$r2Cigar\t$r3Cigar\t$strippedSeq\n";
	print OFILE "$cellBC\t$UMI\t$intBC\t$readCount\t$cigar\t$AS\t$NM\t$r1Cigar\t$r2Cigar\t$r3Cigar\t$seq\t$qname\t$r1Cigar_old\t$r2Cigar_old\t$r3Cigar_old\n";
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
