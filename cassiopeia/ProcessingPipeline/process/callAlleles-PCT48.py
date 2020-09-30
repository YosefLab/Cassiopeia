import re

def call_indels(alignments, ref, output, context = True):

    intBCLim = (21,34)
    intBCLength = 14

    # provide a window in r1/r2/r3 from where cas9 cuts for variation to appear

    r1Cut, r2Cut, r3Cut = 113, 167, 221
    r1Lim = (r1Cut - 12, r1Cut + 12) # ade2
    r2Lim = (r2Cut - 12, r2Cut + 12) # bri1
    r3Lim = (r3Cut - 12, r3Cut + 12) # whtB

    # old limits that just take into account the boundaries of the target site
    #my @r1Lim = (95, 118)
    #my @r2Lim = (149, 172)
    #my @r3Lim = (203, 226)

    refseq = ""
    f = open(ref, "r")
    for i in f:
        if i[0] == ">":
            continue
        else:
            refseq = i
    
    f.close()
    reflength = len(refseq)
    refanchor = refseq[intBCLim[0]-11:intBCLim[0]-1]
    logging.info(f"refseq: {refseq}")
    logging.info(f"refanchor: {refanchor}")

    # open(output, "w")
    # print OFILE "cellBC\tUMI\tintBC\treadCount\tcigar\tAS\tNM\tr1\tr2\tr3\ttargetSite\treadName\tr1+old\tr2+old\tr3+old\n"

    totalReads = 0
    noBCReads = 0
    totalUMIs = 0
    noBCUMIs = 0
    corrected = 0
    seenBadBCs = 0
    correctableUMIs = 0

    alignment_dictionary = {}

    # bamfile = pysam.AlignmentFile(str(alignments), "rb", check_sq = False)

    # out_samfile = pysam.AlignmentFile(str(collapsed_fn), "w", header=bamfile.header)

    for i in alignments.index:
        al = alignments.loc[i]
        readName = al.readName
        seq = al.Seq
        alignmentScore = al.AlignmentScore
        cellBC = al.cellBC
        UMI = al.UMI
        readCount = al.ReadCount
        cigar = al.CIGAR
        start = al.RefStart

        cigarChunks = re.findall(r'\d+[MIDNSHP=X]', cigar)

        refItr = start
        queryItr = al.QueryStart
        queryPad = 0
        printedFlag = 0

        r1Cigar = ""
        r2Cigar = ""
        r3Cigar = ""
        r1Cigar_old = ""
        r2Cigar_old = ""
        r3Cigar_old = ""
        
        r1None = "NC"
        r2None = "NC"
        r3None = "NC"

        intBC = "NC"

        if len(seq) <= 20:
            seenBadBCs += 1
            totalUMIs += 1
            totalReads += readCount
            noBCReads += readCount
            noBCUMIs += 1
            continue

        if len(cigarChunks) > 9:
            continue

        for c in cigarChunks:
            # match => increment iterator
            if re.match(r'\d+M', c):
                matchLen = int(re.match(r'\d+', c).group(0)) # length of match stretch
                refItrN = refItr + matchLen; # the match is in the interval between refItr and refItrN
                # print "refItr refItr refItrN barcode intBCLim[0] intBCLim[1]\n"

                # we're checking if the match goes into the barcode region
                # misses instances where entire bc isn't matched, ie+ there is a 1bp deletion in the BC
                if (intBCLim[0] >= refItr and intBCLim[1] <= refItrN) :
                    # integration BC within seq
                    intBCOffset = intBCLim[0]-refItr
                    if (queryPad > (-1 * intBCLength)) : # entire BC is deleted so reported as "not captured"
                        intBCEndIndex = queryItr+intBCOffset+intBCLength+queryPad
                        intBC = seq[queryItr+intBCOffset:intBCEndIndex]
                        #strippedSeq = substr(seq,intBCEndIndex,length(seq)-intBCEndIndex)
                    
                    #print STDERR "intBCLim[0] refItr:refItr queryItr:queryItr matchLen:matchLen intBCOffset:intBCOffset \n"
                

                if (r1Cut >= refItr and r1Cut <= refItrN and r1Cigar == ""):
                    dist = (r1Cut - refItr)
                    loc = (queryItr + dist)
                    context_l = seq[loc-5:loc]
                    context_r = seq[loc:loc+5]
                    r1None = context_l + "[None]" + context_r
                    #r1Cigar += "None"
                
                if (r2Cut >= refItr and r2Cut <= refItrN and r2Cigar == ""):
                    dist = (r2Cut - refItr)
                    loc = (queryItr + dist)
                    context_l = seq[loc-5:loc]
                    context_r = seq[loc:loc+5]
                    r2None = context_l + "[None]" + context_r
                    #r2Cigar += "None"
                
                if (r3Cut >= refItr and r3Cut <= refItrN and r3Cigar == ""):
                    dist = (r3Cut - refItr)
                    loc = (queryItr + dist)
                    context_l = seq[loc-5:loc]
                    context_r = seq[loc:loc+5]
                    r3None = context_l + "[None]" + context_r
                    #r3Cigar += "None"
                
                # next part is a patch but I think it could be fixed using query pad
                elif (refItr>=intBCLim[0] and refItr<=intBCLim[1] and refItrN>intBCLim[1]): # partial 5' deletion of intBC
                    newIntBCLen = intBCLim[1]-refItr+1
                    intBCEndIndex = queryItr+newIntBCLen
                    intBC = seq[queryItr:intBCEndIndex]
                    #strippedSeq = substr(seq,intBCEndIndex,length(seq)-intBCEndIndex)
                
                elif (refItrN>=intBCLim[0] and refItrN<=intBCLim[1] and refItr<intBCLim[0]): # partial 3' deletion of intBC
                    newIntBCLen = refItrN-intBCLim[0]+1
                    intBCOffset = intBCLim[0]-refItr
                    intBCEndIndex = queryItr+intBCOffset+newIntBCLen
                    intBC = seq[queryItr+intBCOffset:intBCEndIndex]
                    #strippedSeq = substr(seq,intBCEndIndex,length(seq)-intBCEndIndex)
                
                refItr+=matchLen
                queryItr+=matchLen

            elif re.match(r'\d+I', c):
            # insertion in read => add to insertions, increment queryItr
                size = int(re.match(r'\d+', c).group(0))
                if refItr==intBCLim[0]:
                    queryPad = size
                start = queryItr
                queryItr+=size

                # get context of insertion
                context_l = seq[start-5:start]
                context_r = seq[start: start + 5 + size]

                # insertions must lie within one region
                if (refItr>=r1Lim[0] and refItr<=r1Lim[1]):
                    r1Cigar += f"{context_l}[{refItr}:{size}I]{context_r}"
                    r1Cigar_old += f"{refItr}:{size}I"
                
                elif (refItr>=r2Lim[0] and refItr<=r2Lim[1]):
                    r2Cigar += f"{context_l}[{refItr}:{size}I]{context_r}"
                    r2Cigar_old += f"{refItr}:{size}I"
                
                elif (refItr>=r3Lim[0] and refItr<=r3Lim[1]):
                    r3Cigar += f"{context_l}[{refItr}:{size}I]{context_r}"
                    r3Cigar_old += f"{refItr}:{size}I"
                
            
            # deletion in read => add to dels, increment refItr
            elif re.match(r'\d+D', c):
                size = int(re.match(r'\d+', c).group(0))
                refItrLast = refItr
                refItr+=size
                if refIt == intBCLim[0]:
                    queryPad = -1 * size  #deletion starts at intBC

                # get context of deletion
                context_l = seq[queryItr-5:queryItr]
                context_r = seq[queryItr: queryItr + 5]

                # deletions can span multiple regions
                if (r1Lim[0]<=refItr and refItr<=r1Lim[1]) or (r1Lim[0]<=refItrLast and refItrLast<=r1Lim[1]) or (refItrLast<=r1Lim[0] and refItr>=r1Lim[1]) or (refItrLast>=r1Lim[0] and refItr<=r1Lim[1]):
                    r1Cigar+= f"{context_l}[{refItrLast}:{size}D]{context_r}"
                    r1Cigar_old += f"{refItrLast}:{size}D"
                

                if (r2Lim[0]<=refItr and refItr<=r2Lim[1]) or (r2Lim[0]<=refItrLast and refItrLast<=r2Lim[1]) or (refItrLast<=r2Lim[0] and refItr>=r2Lim[1]) or (refItrLast>=r2Lim[0] and refItr<=r2Lim[1]):
                    r2Cigar+=context_l+ f"{context_l}[{refItrLast}:{size}D]{context_r}"
                    r2Cigar_old += f"{refItrLast}:{size}D"

                
                if (r3Lim[0]<=refItr and refItr<=r3Lim[1]) or (r3Lim[0]<=refItrLast and refItrLast<=r3Lim[1]) or (refItrLast<=r3Lim[0] and refItr>=r3Lim[1]) or (refItrLast>=r3Lim[0] and refItr<=r3Lim[1]):
                    r3Cigar+=context_l+ f"{context_l}[{refItrLast}:{size}D]{context_r}"
                    r3Cigar_old += f"{refItrLast}:{size}D"

                
            
            elif re.match(r'\d+H', c):  # hard clipping, does not appear in seq
                # hard clipped, do nothing for now
                size = int(re.match(r'\d+', c).group(0))

                if size > 1:
				    continue

                logging.info(f"HardClip! {qname} {cigar} {seq} {readCount}")
                queryItr += size
            
            else:
                logging.info(f"Unknown CIGAR Occurance: {c} {qname}")
            

        if intBC == "NC" or len(intBC) < intBCLength:
            logging.info(">>> INTBC CORRECTION ATTEMPT! <<<")
            anchor = seq[intBCLim[0]-11:intBCLim[0]-1]
        

            logging.info(f"Cell: {cellBC}")
            logging.info(f"UMI: {UMI}")
            logging.info(f"Sequence anchor: {anchor}")
            logging.info(f"Cigar: {cigar}")
            logging.info(f"Sequence: {seq}")
            logging.info(f"r1: {r1Cigar}")
            logging.info(f"r2: {r2Cigar}")
            logging.info(f"r3: {r3Cigar}")
            logging.info(f"numReads: {readCount}")
            logging.info(f"Alignment Score: {alignmentScore}")

            if anchor == refanchor:
                oldBC = intBC
                intBC = seq[intBCLim[0]-1:intBCLim[0] + intBCLength - 1]
                logging.info(f"Corrected intBC: {intBC} from {oldBC} \n")
                corrected += 1
            
            seenBadBCs+=1
            correctableUMIs+=1
        

        if r1Cigar == "":
            r1Cigar=r1None 
        if r2Cigar == "":
            r2Cigar=r2None 
        if r3Cigar == "":
            r3Cigar=r3None
        if r1Cigar_old == "":
            r1Cigar_old = "None" 
        if r2Cigar_old == "":
            r2Cigar_old = "None"
        if r3Cigar_old == "":
            r3Cigar_old = "None" 

        totalReads+=readCount
        totalUMIs+=1
        if intBC == "NC" or len(intBC) < intBCLength:
            noBCReads+=readCount
            noBCUMIs+=1
            continue
        

        allele = f"{intBC}-{r1Cigar};{r2Cigar};{r3Cigar}"

        # URStr = f"UR:Z:{UMI}"
        # ARStr = f"BC:Z:{cellBC}-{intBC}"
        COStr = ""
        if context:
            COStr = f"{intBC};{r1Cigar};{r2Cigar};{r3Cigar};{r1Cigar_old};{r2Cigar_old};{r3Cigar_old}"
        else:
            COStr = f"{intBC};{r1Cigar_old};{r2Cigar_old};{r3Cigar_old};{r1Cigar_old};{r2Cigar_old};{r3Cigar_old}"
        
        if len(intBC) == intBCLength:
            # print OFILE "qname\tflag\trname\tpos\tmapq\tcigar\trnext\tpnext\ttlen\tseq\tqual\tASStr\tNMStr\tURStr\tARStr\tCOStr\n"
            alignment_dictionary[readName] = (
                cellBC,
                UMI,
                readCount,
                cigar,
                alignmentScore,
                seq,
                intBC,
                COStr
            )
            
    alignment_df = pd.DataFrame.from_dict(alignment_dictionary, orient="index")
    alignment_df.columns = [
        "cellBC",
        "UMI",
        "ReadCount",
        "CIGAR",
        "AlignmentScore",
        "Seq",
        "intBC",
        "Context"
    ]
    alignment_df.index.name = 'readName'
    alignment_df.reset_index(inplace=True)

    propNoBCs = noBCReads/totalReads*100
    propNoBCsUMI = noBCUMIs/totalUMIs*100
    propCorrected = 0
    propCorrectedOverall = 0
    if  seenBadBCs > 0:
        propCorrectedOverall = corrected/(seenBadBCs)*100
        propCorrected = corrected / correctableUMIs*100

    logging.info(f"totalReads: {totalReads} noBCReads: {noBCReads} (" + "{:.1f}".format(propNoBCs) + "%) totalUMIs: {totalUMIs} noBCUMIs: {noBCUMIs} (" + "{:.1f}".format(propNoBCsUMI) + "%)")
    logging.info(f"Correctable intBCs: ({correctableUMIs}) of {totalUMIs} (" + "{:.1f}".format((correctableUMIs / totalUMIs)*100) + "%)")
    logging.info(f"Corrected intBCs: {corrected} of {correctableUMIs} (" + "{:.1f}".format(propCorrected) + "% of correctable, " + "{:.1f}".format(propCorrectedOverall) + "% overall)")

    return alignment_df