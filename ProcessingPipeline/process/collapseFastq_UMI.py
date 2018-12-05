import numpy as np
import sys
import argparse

from Sequencing import fastq

def collapse_fastq(reads,outfile):

	counter=0

	collapsedReads = {} # dict of unique reads; collapsedRead{seq} = [read,count]
	for r in reads: #itertools.islice(reads,10000): # iterate thru reads and collapse as necessary

		if counter%1000000 == 0:
			print str(counter) + " reads processed ..."

		n = r.name.split('_')
		nseq = n[1] + "_" + n[2] + "_" + r.seq #[rSlice]

		if nseq in collapsedReads: # collapsable sequence
			[oldRead,count] = collapsedReads[nseq]
			# maximize quality
			nqual = r.qual #[rSlice]
			fqualList = []
			for oR_q,nR_q in zip(oldRead.qual,nqual):
				if oR_q > nR_q:
					fqualList.append(oR_q)
				else:
					fqualList.append(nR_q)
			fqual = ''.join(fqualList)
			count = count+1
			nRead = fastq.Read(oldRead.name,r.seq,fqual)
			collapsedReads[nseq] = [nRead,count]
		else:
			nRead = fastq.Read(r.name,r.seq,r.qual) #[rSlice])
			collapsedReads[nseq] = [nRead,1]
		counter = counter + 1

	fh = open(outfile, 'w')
	for i in collapsedReads:
		[r,count] = collapsedReads[i]
		#n = r.name.split(' ')
		
		fh.write(str(fastq.Read(r.name + "_" + str(count),r.seq,r.qual)))
	fh.close()

if __name__ == '__main__':
	import itertools
	parser = argparse.ArgumentParser()
	parser.add_argument('R1', help='input Reads fastq file name (can ge gzip\'ed)')
	parser.add_argument('outfileCollapsed', help='output fastq of collapsed reads')

	args = parser.parse_args()
	reads = fastq.reads(args.R1)

	collapse_fastq(reads,args.outfileCollapsed)

