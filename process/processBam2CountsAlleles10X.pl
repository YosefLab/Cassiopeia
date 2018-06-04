#!/usr/bin/env perl

### Update notes
#
# 2017/09/08: no indels allowed before intBC read
#             report Missing instead of none of alignment ends before tS
#
#

use strict;
use warnings;

die "\nuse: $0 bamFile outfile.txt\n\n" if $#ARGV!=1;

my $samtool = "samtools";
my $bamfile = $ARGV[0];
my $outfile = $ARGV[1];

my @intBCLim = (27,34);
my $intBCLength = 8;

my @r1Lim = (133,155); # bao_ade2
my @r2Lim = (86,109); # dros_bam3
my @r3Lim = (40,62); # whiteB

open(OFILE,">$outfile");
print OFILE "cellBC\tUMI\tintBC\treadCount\tcigar\tAS\tNM\tr1\tr2\tr3\ttargetSite\treadName\n";

my $totalReads = 0;
my $noBCReads = 0;
my $totalUMIs = 0;
my $noBCUMIs = 0;
open(IN,"$samtool view -F 4 $bamfile |") or die($!);
while(<IN>) {
	chomp;
	my ($qname,$flag,$rname,$pos,$mapq,$cigar,$rnext,$pnext,$tlen,$seq,$qual,$ASStr,$NMStr) = split(/\t/,$_);
	my ($cellBC,$UMI,$readCount) = split(/_/,$qname);
	my $start = $pos;
	my @cigarChunks = $cigar =~ m/\d+[MIDNSHP=X]/g;
	my ($t,$i,$AS) = split(/:/,$ASStr);
	($t,$i,my $NM) = split(/:/,$NMStr);

	my $refItr=$start;
	my $queryItr=0;
	my $queryPad = 0;
	my $printedFlag = 0;

	my $r1Cigar="";
	my $r2Cigar="";
	my $r3Cigar="";
	my $intBC="NC";

	foreach my $c (@cigarChunks) {
		# match => increment iterator
		if ($c =~ m/(\d+)M/) {
			my $matchLen = $1;
			my $refItrN = $refItr+$matchLen;
			# misses instances where entire bc isn't matched, ie. there is a 1bp deletion in the BC
			if ($intBCLim[0]>=$refItr && $intBCLim[1]<=$refItrN) {
				# integration BC within seq
				my $intBCOffset = $intBCLim[0]-$refItr;
				if ($queryPad>(-1 * $intBCLength)) { # entire BC is deleted so reported as "not captured"
					$intBC = substr($seq,$queryItr+$intBCOffset,$intBCLength+$queryPad);
					my $intBCEndIndex = $queryItr+$intBCOffset+$intBCLength+$queryPad;
				}
				#print STDERR "$intBCLim[0] refItr:$refItr queryItr:$queryItr matchLen:$matchLen intBCOffset:$intBCOffset \n";
			}
			# next part is a patch but I think it could be fixed using query pad
			elsif ($refItr>=$intBCLim[0] && $refItr<=$intBCLim[1] && $refItrN>$intBCLim[1]) { # partial 5' deletion of intBC
				my $newIntBCLen = $intBCLim[1]-$refItr+1;
				#$intBC = substr($seq,$queryItr,$newIntBCLen);
				$intBC = "Bad"; # 5' deletion not permitted
				my $intBCEndIndex = $queryItr+$newIntBCLen;
			}
			elsif ($refItrN>=$intBCLim[0] && $refItrN<=$intBCLim[1] && $refItr<$intBCLim[0]) { # partial 3' deletion of intBC
				my $newIntBCLen = $refItrN-$intBCLim[0]+1;
				my $intBCOffset = $intBCLim[0]-$refItr;
				$intBC = substr($seq,$queryItr+$intBCOffset,$newIntBCLen);
				my $intBCEndIndex = $queryItr+$intBCOffset+$newIntBCLen;
			} 
			$refItr+=$matchLen;
			$queryItr+=$matchLen;
		} elsif ($c =~ m/(\d+)I/) {
		# insertion in read => add to insertions, increment itr
			my $size=$1;
			$queryPad = $size if $refItr==$intBCLim[0];
			$queryItr+=$size;
			# insertions must lie within one region
			if ($refItr>=$r1Lim[0] && $refItr<=$r1Lim[1]) {
				$r1Cigar.="${refItr}:${size}I";
			}
			elsif ($refItr>=$r2Lim[0] && $refItr<=$r2Lim[1]) {
				$r2Cigar.="${refItr}:${size}I";
			}
			elsif ($refItr>=$r3Lim[0] && $refItr<=$r3Lim[1]) {
				$r3Cigar.="${refItr}:${size}I";
			}
		}
		# deletion in read => add to dels, increment refItr
		elsif ($c =~ m/(\d+)D/) {
			my $size=$1;
			my $refItrLast=$refItr;
			$refItr+=$size;
			$queryPad = -1 * $size if $refItr==$intBCLim[0];
			# deletions can span multiple regions
			if (($r1Lim[0]<=$refItr && $refItr<=$r1Lim[1]) ||
			   ($r1Lim[0]<=$refItrLast && $refItrLast<=$r1Lim[1]) ||
			   ($refItrLast<=$r1Lim[0] && $refItr>=$r1Lim[1]) ||
			   ($refItrLast>=$r1Lim[0] && $refItr<=$r1Lim[1])) {
				$r1Cigar.="${refItrLast}:${size}D";
			}

			if (($r2Lim[0]<=$refItr && $refItr<=$r2Lim[1]) ||
			   ($r2Lim[0]<=$refItrLast && $refItrLast<=$r2Lim[1]) ||
			   ($refItrLast<=$r2Lim[0] && $refItr>=$r2Lim[1]) ||
			   ($refItrLast>=$r2Lim[0] && $refItr<=$r2Lim[1])) {
				$r2Cigar.="${refItrLast}:${size}D";
			}

			if (($r3Lim[0]<=$refItr && $refItr<=$r3Lim[1]) ||
			   ($r3Lim[0]<=$refItrLast && $refItrLast<=$r3Lim[1]) ||
			   ($refItrLast<=$r3Lim[0] && $refItr>=$r3Lim[1]) ||
			   ($refItrLast>=$r3Lim[0] && $refItr<=$r3Lim[1])) {
				$r3Cigar.="${refItrLast}:${size}D";
			}
		}
		elsif ($c =~ m/(\d+)H/) { # hard clipping, does not appear in seq
			# hard clipped, do nothing for now
			my $size = $1;
			#print STDERR "$qname $cigar $pos $seq $size\n";
		}
		elsif ($c =~ m/\d+[NSP=X]/) {
			print STDERR "Unknown CIGAR Occurance: $c $qname\n";
		}
	}
	$r3Cigar="Missing" if $refItr<$r3Lim[0]; # this assumes intBC before cut sites
	$r2Cigar="Missing" if $refItr<$r2Lim[0];
	$r1Cigar="Missing" if $refItr<$r1Lim[0];
	$intBC="NC" if $intBC =~ m/^\s*$/; # patch

	$r1Cigar="None" if $r1Cigar eq "";
	$r2Cigar="None" if $r2Cigar eq "";
	$r3Cigar="None" if $r3Cigar eq "";
	#print OFILE "$cellBC\t$UMI\t$intBC\t$readCount\t$r1Cigar\t$r2Cigar\t$r3Cigar\t$strippedSeq\n";
	print OFILE "$cellBC\t$UMI\t$intBC\t$readCount\t$cigar\t$AS\t$NM\t$r1Cigar\t$r2Cigar\t$r3Cigar\t$seq\t$qname\n";
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
print "totalReads: $totalReads noBCReads: $noBCReads (" . sprintf("%.1f",$propNoBCs) . "%) ";
print "totalUMIs: $totalUMIs noBCUMIs: $noBCUMIs (" . sprintf("%.1f",$propNoBCsUMI) . "%)\n";

