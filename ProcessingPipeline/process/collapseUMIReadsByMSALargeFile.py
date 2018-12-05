import sys
import argparse
import Bio.AlignIO
import numpy as np
import re
import pandas as pd
import time
import matplotlib.pyplot as plt
# Import Jeff's Sequencing package
sys.path.append("/home/jah/projects/sequencing/code")
from Sequencing import fastq

from subprocess import Popen, PIPE, STDOUT
from Bio.Align import AlignInfo
from io import StringIO

def get_consensus(seqs,counts):
# followed advice from here:
#  http://stackoverflow.com/questions/18860962/run-clustalw2-without-input-fasta-file

	clustalo = '/home/mchan/software/clustalo-1.2.3-Ubuntu-x86_64'

	# Multiple Sequence Alignment
	s1 = [] # reads expanded to represent accurate counts
	for seq,readCount in zip(seqs,counts):
		t1 = [seq] * readCount
		s1 = s1 + t1
	s2 = ['>seq'+str(i)+'\n'+s1[i]+'\n' for i in range(len(s1))]
	str1 = ''.join(s2)
	proc = Popen([clustalo, '--infile=-', '--threads=4'],stdout=PIPE, stdin=PIPE, stderr=STDOUT)
	stdout = proc.communicate(input=str1.encode())[0]

	alignment = Bio.AlignIO.read(StringIO(stdout.decode()),'fasta')
	summary_align = AlignInfo.SummaryInfo(alignment)
	consensusSeq = summary_align.gap_consensus(threshold=0.6,ambiguous='N')
	consensusSeq = str(consensusSeq).replace("-","")
	consensusSeq = re.sub('N+$','',str(consensusSeq)) # trim trailing Ns
	return str(consensusSeq)

def collapseUMIs(reads, readThres, outfile):
# collpase reads assuming cellBC and UMI are true
# identifies consensus sequence and reports as sequence for each cellBC-UMI combination

	numReadsQualFilt = 0

	UMIGrps = {}
	for r in reads: # itertools.islice(reads,10000):
		avgQ50 = np.mean(fastq.decode_sanger(r.qual[0:50]))
		if avgQ50<20:
			numReadsQualFilt = numReadsQualFilt+1
			continue
		n = r.name.split('_')
		cellGroup = n[1] + "_" + n[2]
		readcount = int(n[3])

		if cellGroup in UMIGrps:
			[seqs,counts] = UMIGrps[cellGroup]
			seqs.append(r.seq)
			counts.append(readcount)
		else:
			UMIGrps[cellGroup] = [[r.seq],[readcount]]

	print("# of cell-UMI groups: " + str(len(UMIGrps)) + " (includes <" + str(readThres) + ")")

	read_dist = []
	for u in UMIGrps:
		[seqs, counts] = UMIGrps[u]
		read_dist.append(sum(counts))


	h = plt.figure(figsize=(14, 10))
	ax = plt.hist(read_dist, log=True)
	plt.ylabel("Frequency")
	plt.xlabel("Number of Reads")
	plt.title("Reads Per UMI")
	plt.savefig("collapsedUMIs_reads_per_umi.init.png")
	plt.close()

	readThresh = np.percentile(read_dist, 99) / 10
	print("Filtering out UMIs with less than " + str(readThresh) + " reads")

	fh = open(outfile,'w')
	fh.write("cellBC\tUMI\treadCount\tconsensusSeq\n")

	numBelowReadThres = 0
	numMaj = 0
	numCon=0
	numSingles = 0
	counter = 1
	for k in UMIGrps: # each UMI group consists of reads from the same molecule

		[seqs,counts] = UMIGrps[k]
		grpSize = sum(counts)
		if grpSize < readThres: # too few reads to include
			numBelowReadThres = numBelowReadThres + 1
			continue

		n = k.split("_")
		if len(seqs)==1: # trivial case added 9/11/2017
			numSingles = numSingles+1
			fh.write("\t".join([str(n[0]),str(n[1]),str(counts[0]),seqs[0]]) + "\n")
		else:
			#
			# Update 9/1/2017: try to improve speeds by increasing the number of same reads to feed
			#	into majority instead of consensus finding
			#   trim to length of 25th percentile read ranked by length
			#
			s1 = pd.DataFrame({"seq": seqs, "readCount": counts})
			s1["seqLen"] = s1["seq"].str.len()
			s1 = s1.sort_values("seqLen").reset_index(drop=True) # sorts reads by length in ascending
			totalReads = s1["readCount"].sum()
			cReads = s1["readCount"].cumsum() # cumulative
			rPctile = 0.3*totalReads # 30th percentile
			rPctileIndex = cReads[cReads>=rPctile].index[0] # index of seq length
			sLen = s1.loc[rPctileIndex,"seqLen"]
			s1["seq"] = s1["seq"].str[0:sLen]
			s2 = s1.groupby(["seq"]).agg({"readCount": np.sum}).sort_values("readCount",ascending=False) # indexed by seq

			grpProp = s2.loc[s2.index[0],"readCount"]/float(totalReads)

			if grpProp>.50:
				consensusSeq = s2.index[0]
				numMaj = numMaj+1
			else:
				consensusSeq = get_consensus(s2.index.tolist(),s2["readCount"].tolist())
				numCon = numCon+1

			# print Entry
			fh.write("\t".join([str(n[0]),str(n[1]),str(totalReads),consensusSeq]) + "\n")

		counter = counter+1
		if counter%1000==0:
			print(str(counter) + " groups processed...")

	fh.close()

	print("# of cell-UMI groups = " + str(len(UMIGrps)))
	print("# reads qual <20 (filtered) = " + str(numReadsQualFilt))
	print("# grps w/ reads<" + str(readThres) + " = " + str(numBelowReadThres))
	print("# grps singles = " + str(numSingles))
	print("# grps >0.5 = " + str(numMaj))
	print("# grps concensus = " + str(numCon))


if __name__ == '__main__':

	t0 = time.time()
	parser = argparse.ArgumentParser()
	parser.add_argument('fq', help='fastq of reads collapsed by sequence')
	parser.add_argument('readThres', help='UMIs with <readThres will be thrown out; default=3', default=3)
	parser.add_argument('outfile', help='collapsedFastqTable.txt')

	args = parser.parse_args()

	reads = fastq.reads(args.fq)

	collapseUMIs(reads, int(args.readThres), args.outfile)
	print("Final Time: " + str(time.time() - t0))
