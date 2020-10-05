#!/usr/bin/env perl

use strict;
use warnings;

die "\nuse: $0 bamFile ref.fa outfile.txt contextopt\n\n" if $#ARGV!=3;

# Note: Stripped seqs need to be fixed depending on direction of intBC

my $samtool = "samtools";
my $bamfile = $ARGV[0];
my $reffile = $ARGV[1]; # reference alignment for intBC calling
my $outfile = $ARGV[2];
my $contextarg = $ARGV[3];

my $contextopt = 1;
if ($contextarg eq "--nocontext") {
	$contextopt = 0;
}
	

my @intBCLim = (21,34);
my $intBCLength = 14;
my @intBCs;

# provide a window in r1/r2/r3 from where cas9 cuts for variation to appear

my ($r1Cut, $r2Cut, $r3Cut) = (113, 167, 221);
my @r1Lim = ($r1Cut - 12, $r1Cut + 12); # ade2
my @r2Lim = ($r2Cut - 12, $r2Cut + 12); # bri1
my @r3Lim = ($r3Cut - 12, $r3Cut + 12); # whtB

# old limits that just take into account the boundaries of the target site
#my @r1Lim = (95, 118);
#my @r2Lim = (149, 172);
#my @r3Lim = (203, 226);

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
my $refanchor = substr($ref, $intBCLim[0]-11, 10);
print "refseq: $ref\n";
print "refanchor: $refanchor\n";

open(OFILE,">$outfile");
#print OFILE "cellBC\tUMI\tintBC\treadCount\tcigar\tAS\tNM\tr1\tr2\tr3\ttargetSite\treadName\tr1.old\tr2.old\tr3.old\n";

my $totalReads = 0;
my $noBCReads = 0;
my $totalUMIs = 0;
my $noBCUMIs = 0;
my $corrected = 0;
my $seenBadBCs = 0;
my $correctableUMIs = 0;

# COPY HEADER
open(IN, "$samtool view -H $bamfile|");
while (<IN>) {
	chomp;
	print OFILE "$_\n";
}

close(IN);

open(IN,"$samtool view -F 4 $bamfile |") or die($!);
LINE: while(<IN>) {
	chomp;
	my ($qname,$flag,$rname,$pos,$mapq,$cigar,$rnext,$pnext,$tlen,$seq,$qual,$ASStr,$NMStr) = split(/\t/,$_);
	my ($cellBC,$UMI,$readCount) = split(/_/,$qname);
	my $start = $pos;
	$cigar = "13M";
	my @cigarChunks = $cigar =~ m/\d+[MIDNSHP=X]/g;
	print "CC: @cigarChunks\n";
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
	
	my $r1None = "NC";
	my $r2None = "NC";
	my $r3None = "NC";

	my $intBC="NC";
	#my $strippedSeq = $seq;
	#

	if ((length $seq) <= 0) {
		$seenBadBCs++;
		$totalUMIs++;
		$totalReads += $readCount;
		$noBCReads += $readCount;
		$noBCUMIs++;
		next LINE;
	}

	if ((scalar @cigarChunks) > 9) { 
		next LINE;
	}

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
					#$strippedSeq = substr($seq,$intBCEndIndex,length($seq)-$intBCEndIndex);
				}
				#print STDERR "$intBCLim[0] refItr:$refItr queryItr:$queryItr matchLen:$matchLen intBCOffset:$intBCOffset \n";
			}

			if ($r1Cut >= $refItr && $r1Cut <= $refItrN && $r1Cigar eq ""){
				my $dist = ($r1Cut - $refItr);
				my $loc = ($queryItr + $dist);
				my $context_l = substr($seq, $loc-5, 5);
				my $context_r = substr($seq, $loc, 5);
				$r1None = "${context_l}[None]${context_r}";
				#$r1Cigar .= "None";
			}
			if ($r2Cut >= $refItr && $r2Cut <= $refItrN && $r2Cigar eq ""){
				my $dist = ($r2Cut - $refItr);
				my $loc = ($queryItr + $dist);
				my $context_l = substr($seq, $loc-5, 5);
				my $context_r = substr($seq, $loc, 5);
				$r2None = "${context_l}[None]${context_r}";
				#$r2Cigar .= "None";
			}
			if ($r3Cut >= $refItr && $r3Cut <= $refItrN && $r3Cigar eq ""){
				my $dist = ($r3Cut - $refItr);
				my $loc = ($queryItr + $dist);
				my $context_l = substr($seq, $loc-5, 5);
				my $context_r = substr($seq, $loc, 5);
				$r3None = "${context_l}[None]${context_r}";
				#$r3Cigar .= "None";
			}


			# next part is a patch but I think it could be fixed using query pad
			elsif ($refItr>=$intBCLim[0] && $refItr<=$intBCLim[1] && $refItrN>$intBCLim[1]) { # partial 5' deletion of intBC
				my $newIntBCLen = $intBCLim[1]-$refItr+1;
				$intBC = substr($seq,$queryItr,$newIntBCLen);
				my $intBCEndIndex = $queryItr+$newIntBCLen;
				#$strippedSeq = substr($seq,$intBCEndIndex,length($seq)-$intBCEndIndex);
			}
			elsif ($refItrN>=$intBCLim[0] && $refItrN<=$intBCLim[1] && $refItr<$intBCLim[0]) { # partial 3' deletion of intBC
				my $newIntBCLen = $refItrN-$intBCLim[0]+1;
				my $intBCOffset = $intBCLim[0]-$refItr;
				$intBC = substr($seq,$queryItr+$intBCOffset,$newIntBCLen);
				my $intBCEndIndex = $queryItr+$intBCOffset+$newIntBCLen;
				#$strippedSeq = substr($seq,$intBCEndIndex,length($seq)-$intBCEndIndex);
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
			my $context_l = substr($seq, $start-5, 5);
			my $context_r = substr($seq, $start, 5 + $size);

			# insertions must lie within one region
			if ($refItr>=$r1Lim[0] && $refItr<=$r1Lim[1]) {
				$r1Cigar.="${context_l}[${refItr}:${size}I]${context_r}";
				$r1Cigar_old .= "${refItr}:${size}I";
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
			my $context_l = substr($seq, $queryItr-5, 5);
			my $context_r = substr($seq, $queryItr, 5);

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
			print STDERR "HardClip! $qname $cigar $pos $seq $size $readCount\n";

			if ($size > 1) { 
				next LINE;
			}

			$queryItr += $size;
		}
		elsif ($c =~ m/\d+[NSP=X]/) {
			print STDERR "Unknown CIGAR Occurance: $c $qname\n";
		}
	}

	if ($intBC eq "NC" || (length $intBC) < $intBCLength) {
		print ">>> INTBC CORRECTION ATTEMPT! <<<\n";
		my $anchor = substr($seq, $intBCLim[0]-11, 10);
	

		print "Cell: $cellBC\n";
		print "UMI: $UMI\n";
		print "Sequence anchor: $anchor\n";
		print "Cigar: $cigar\n";
		print "Sequence: $seq\n";
		print "r1: $r1Cigar\n";
		print "r2: $r2Cigar\n";
		print "r3: $r3Cigar\n";
		print "numReads: $readCount\n";
		print "AS: $AS\n";

		if ($anchor eq $refanchor) {
			my $oldBC = $intBC;
			$intBC = substr($seq, $intBCLim[0]-1, $intBCLength);
			print "Corrected intBC: $intBC\tFrom $oldBC \n";
			$corrected++;
		}
		$seenBadBCs++;
		$correctableUMIs++
	}

	$r1Cigar=$r1None if $r1Cigar eq "";
	$r2Cigar=$r2None if $r2Cigar eq "";
	$r3Cigar=$r3None if $r3Cigar eq "";
	$r1Cigar_old = "None" if $r1Cigar_old eq "";
	$r2Cigar_old = "None" if $r2Cigar_old eq "";
	$r3Cigar_old = "None" if $r3Cigar_old eq "";

	$totalReads+=$readCount;
	$totalUMIs+=1;
	if ($intBC eq "NC" || (length $intBC) < $intBCLength) {
		$noBCReads+=$readCount;
		$noBCUMIs+=1;
		next;
	}

	my $allele = "${intBC}-${r1Cigar};${r2Cigar};${r3Cigar}";

	my $URStr = "UR:Z:${UMI}";
	my $ARStr = "BC:Z:${cellBC}-${intBC}";
	my $COStr = "";
	if ($contextopt) {
		$COStr = "CO:Z:${intBC};${r1Cigar};${r2Cigar};${r3Cigar};${r1Cigar_old};${r2Cigar_old};${r3Cigar_old}";
	} else {
		$COStr = "CO:Z:${intBC};${r1Cigar_old};${r2Cigar_old};${r3Cigar_old};${r1Cigar_old};${r2Cigar_old};${r3Cigar_old}";
	}
	if (length $intBC == $intBCLength) { 
		print OFILE "$qname\t$flag\t$rname\t$pos\t$mapq\t$cigar\t$rnext\t$pnext\t$tlen\t$seq\t$qual\t$ASStr\t$NMStr\t$URStr\t$ARStr\t$COStr\n";
	}

}


close IN;
close OFILE;

my $propNoBCs = $noBCReads/$totalReads*100;
my $propNoBCsUMI = $noBCUMIs/$totalUMIs*100;
my $propCorrected = 0;
my $propCorrectedOverall = 0;
if  ($seenBadBCs > 0) {
	$propCorrectedOverall = $corrected/($seenBadBCs)*100;
	$propCorrected = $corrected / $correctableUMIs*100; 
}
print "totalReads: $totalReads noBCReads: $noBCReads (" . sprintf("%.1f",$propNoBCs) . "%) ";
print "totalUMIs: $totalUMIs noBCUMIs: $noBCUMIs (" . sprintf("%.1f",$propNoBCsUMI) . "%)\n";
print "Correctable intBCs: ($correctableUMIs) of $totalUMIs (" . sprintf("%.1f", , ($correctableUMIs) / $totalUMIs*100) . "%)\n";
print "Corrected intBCs: $corrected of $correctableUMIs (" . sprintf("%.1f",$propCorrected) . "% of correctable, " . sprintf("%.1f", $propCorrectedOverall) . "% Overall)\n";
