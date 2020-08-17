import sys
import argparse
import itertools

def changeCellBCID(alleleTableIN, sampleID, alleleTableOUT):

	fOut = open(alleleTableOUT, 'w')
	header = True
	with open(alleleTableIN, 'r') as umiList:
		for umi in umiList:
			if header:
				fOut.write(umi)
				header = False
				continue
			umiAttr = umi.split("\t")
			cellBC = umiAttr[0]
			new_cellBC = sampleID + "." + cellBC
			fOut.write(new_cellBC)
			for i in range(1,len(umiAttr)):
				fOut.write("\t" + umiAttr[i])
	fOut.close()
	
if __name__ == '__main__':
	
	parser = argparse.ArgumentParser()
	parser.add_argument('alleleTable', help='alleleTable IN')
	parser.add_argument('sampleID', help='sampleID')
	parser.add_argument('out', help='alleleTable OUT')

	args = parser.parse_args()
	
	changeCellBCID(args.alleleTable, args.sampleID, args.out)
