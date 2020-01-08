use strict;
use warnings;

use Bio::SeqReader::Fastq;
use IO::Uncompress::AnyUncompress;

my $r1_handle = new IO::Uncompress::AnyUncompress($ARGV[0]); # R1 file handle
my $r2_handle = new IO::Uncompress::AnyUncompress($ARGV[1]); # R2 file handle
my $wh = $ARGV[2]; # cell bc white list file handle
my $outfile = $ARGV[3]; # output file

my $r1_fastq = new Bio::SeqReader::Fastq(fh => $r1_handle);
my $r2_fastq = new Bio::SeqReader::Fastq(fh => $r2_handle);

my @barcodes;
my @whitelist;

# let's define a quick function for computing hamming distances
sub hd {
    return ($_[0] ^ $_[1]) =~ tr/\001-\255//;
}

# collect all white list elements in an array
open(BC_IN, "< $wh") or die "Couldn't open file $wh";
while (<BC_IN>) {
  chomp;
  push @whitelist, $_;
}
close BC_IN;

# turn the whitelist array into a hash for quick lookup
my %whitelist = map {$_ => 1} @whitelist;

# iterate over r2s and collect CellBC & umiBC information. Do naive error correction
# against a white list of cell ranger barcodes
my $counter = 0;
my $mismatches = 0;
while (my $so = $r1_fastq->next()) {
  # I only want to see the first 100 sequences for testing purposes
  last if $counter == 1000;
  my $s = $so->seq();
  my $cbc = substr $s, 0, 16;

  # Perform error correction if cell barcode isn't in whitelist
  if (!exists($whitelist{$cbc})) {

    $mismatches++;
    # find a cell barcode that's one hamming distance away
    # if we don't find one, then we'll have to leave teh cell barcode for now, to
    # be error corrected later on with its sequence info.
    foreach my $crbc (keys %whitelist) {
      if (hd($crbc, $cbc) == 1) {
        $cbc = $crbc;
        last;
      }
    }

  }

  my $ubc = substr $s, 16, (length $s) - 16;
  push @barcodes, [$cbc, $ubc];
  $counter++;
}
print "$barcodes[0][0]\t$barcodes[0][1]\n";
print "$barcodes[1][0]\t$barcodes[1][1]\n";
print "$mismatches mistmatches of $counter\n";
