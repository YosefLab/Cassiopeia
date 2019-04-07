import numpy as np
import pandas as pd
from Cassiopeia.ProcessingPipeline.process import filterMoleculeTables

def test_umi_errcorr():

    sdata_true = pd.read_csv("test/data/sim_data_true.csv", sep='\t')
    sdata_umierr = pd.read_csv("test/data/sim_data_errumi.csv", sep='\t')
    sdata_true["allele"] = sdata_true.apply(lambda row: row.r1 + row.r2 + row.r3, axis=1)
    sdata_umierr["allele"] = sdata_umierr.apply(lambda row: row.r1 + row.r2 + row.r3, axis=1)

    corrected = filterMoleculeTables.errorCorrectUMI(sdata_umierr, "", verbose=False)

    assert sdata_true.shape[0] == corrected.shape[0]

    # Validate that all UMIs were corrected by looking at readcounts
    sdata_true["gcol"] = sdata_true[["cellBC", "UMI"]].apply(lambda x: '_'.join(x), axis=1)
    corrected["gcol"] = corrected[["cellBC", "UMI"]].apply(lambda x: '_'.join(x), axis=1)

    # First test that we have the same UMIs as the true dataset
    assert sdata_true["gcol"].equals(corrected["gcol"])

    for i in sdata_true.index:

        gcol = sdata_true.loc[i, "gcol"]
        rc_true = sdata_true.loc[i, "readCount"]

        rc_corr = corrected.loc[corrected["gcol"] == gcol, "readCount"].iloc[0]

        assert rc_corr == rc_true


def test_ibc_errcorr():

    sdata_true = pd.read_csv("test/data/sim_data_true.csv", sep='\t')
    sdata_ibcerr = pd.read_csv("test/data/sim_data_erribc.csv", sep='\t')

    sdata_true["allele"] = sdata_true.apply(lambda row: row.r1 + row.r2 + row.r3, axis=1)
    sdata_ibcerr["allele"] = sdata_ibcerr.apply(lambda row: row.r1 + row.r2 + row.r3, axis=1)

    corrected = filterMoleculeTables.errorCorrectIntBC(sdata_ibcerr, "", verbose=False)

    assert sdata_true.shape == corrected.shape

    # Test that intBCs have the same number of UMIs in each cell
    sdata_true["gcol"] = sdata_true[["cellBC", "intBC"]].apply(lambda x: '_'.join(x), axis=1)
    corrected["gcol"] = corrected[["cellBC", "intBC"]].apply(lambda x: '_'.join(x), axis=1)

    # Make sure we have the same cell barcode / integration barcode combination
    assert sdata_true["gcol"].equals(corrected["gcol"])

    for i in sdata_true["gcol"].unique():

        true_vals = sdata_true.loc[(sdata_true["gcol"] == i),:].groupby(["intBC"]).agg({"UMI": "count"})
        corr_vals = corrected.loc[(corrected["gcol"] == i), :].groupby(["intBC"]).agg({"UMI": "count"})

        assert true_vals["UMI"].equals(corr_vals["UMI"])

def test_allele_corr():

    sdata_true = pd.read_csv("test/data/sim_data_true.csv", sep='\t')
    sdata_alleles = pd.read_csv("test/data/sim_data_erralleles.csv", sep='\t')

    sdata_true["allele"] = sdata_true.apply(lambda row: row.r1 + row.r2 + row.r3, axis=1)
    sdata_alleles["allele"] = sdata_alleles.apply(lambda row: row.r1 + row.r2 + row.r3, axis=1)

    corrected = filterMoleculeTables.pickAlleles(sdata_alleles, "", verbose=False)

    assert sdata_true.shape == corrected.shape

    # Let's make sure that each integration barcode maps to the same allele across
    # the two datasets

    sdata_true["gcol"] = sdata_true[["cellBC", "intBC"]].apply(lambda x: '_'.join(x), axis=1)
    corrected["gcol"] = corrected[["cellBC", "intBC"]].apply(lambda x: '_'.join(x), axis=1)

    assert sdata_true["gcol"].equals(corrected["gcol"])

    for i in sdata_true["gcol"].unique():

        true_vals = sdata_true.loc[(sdata_true["gcol"] == i),:].groupby(["intBC"]).agg({"allele": "unique"})
        corr_vals = corrected.loc[(corrected["gcol"] == i), :].groupby(["intBC"]).agg({"allele": "unique"})

        # make sure they map equally
        assert true_vals["allele"].equals(corr_vals["allele"])

        # make sure that each intBC only maps to a single allele
        assert true_vals.shape[0] == 1
        assert corr_vals.shape[0] == 1
