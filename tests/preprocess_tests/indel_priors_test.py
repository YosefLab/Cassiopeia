"""Tests for pp.compute_indel_priors on allele tables."""

import unittest

import pandas as pd

import cassiopeia as cas


class TestIndelPriors(unittest.TestCase):
    def setUp(self):
        at_dict = {
            "cellBC": ["cellA", "cellA", "cellA", "cellB", "cellC"],
            "intBC": ["A", "B", "C", "A", "C"],
            "r1": ["None", "ATC", "GGG", "None", "GAA"],
            "r2": ["None", "AAA", "GAA", "None", "GAA"],
            "r3": ["ATC", "TTT", "ATA", "ATA", "ATA"],
            "UMI": [5, 10, 1, 30, 30],
        }

        self.alleletable_basic = pd.DataFrame.from_dict(at_dict)

        self.mutation_priors = pd.DataFrame.from_dict(
            {
                "ATC": 0.5,
                "GGG": 0.2,
                "GAA": 0.1,
                "AAA": 0.05,
                "TTT": 0.05,
                "ATA": 0.1,
            },
            orient="index",
            columns=["freq"],
        )

        ## setup complicated allele table
        at_dict = {
            "cellBC": [
                "cellA",
                "cellA",
                "cellA",
                "cellB",
                "cellB",
                "cellC",
                "cellD",
                "cellD",
            ],
            "intBC": ["A", "B", "C", "B", "C", "A", "A", "B"],
            "r1": ["AAA", "AAB", "AAC", "AAD", "ABA", "ABB", "AAA", "AAB"],
            "r2": ["BAA", "BAB", "BAC", "BAD", "BBA", "BBB", "BAA", "BAB"],
            "r3": ["CAA", "CAB", "CAC", "CAD", "CBA", "CBB", "CAA", "CAB"],
            "UMI": [5, 10, 30, 30, 10, 10, 3, 3],
            "Mouse": ["M1", "M1", "M1", "M1", "M1", "M1", "M2", "M2"],
        }
        self.allele_table_mouse = pd.DataFrame(at_dict)

        ## set up non-cassiopeia allele table
        self.noncassiopeia_alleletable = self.alleletable_basic.copy()
        self.noncassiopeia_alleletable.rename(
            columns={"r1": "cs1", "r2": "cs2", "r3": "cs3"}, inplace=True
        )

        # allele table with conflicts
        at_dict = {
            "cellBC": ["cellA", "cellA", "cellA", "cellB", "cellC", "cellA"],
            "intBC": ["A", "B", "C", "A", "C", "A"],
            "r1": ["None", "ATC", "GGG", "None", "GAA", "None"],
            "r2": ["None", "AAA", "GAA", "None", "GAA", "ACT"],
            "r3": ["ATC", "TTT", "ATA", "ATA", "ATA", "None"],
            "UMI": [5, 10, 1, 30, 30, 5],
        }
        self.alleletable_conflict = pd.DataFrame.from_dict(at_dict)

    def test_compute_indel_priors(self):
        indel_probabilities = cas.pp.compute_indel_priors(self.alleletable_basic, per_site=False)

        expected_priors = pd.DataFrame.from_dict(
            {
                "GGG": [1, 1 / 3],
                "GAA": [1, 1 / 3],
                "ATA": [2, 2 / 3],
                "ATC": [2, 2 / 3],
                "AAA": [1, 1 / 3],
                "TTT": [1, 1 / 3],
            },
            orient="index",
            columns=["count", "freq"],
        )

        for indel in expected_priors.index:
            self.assertIn(indel, indel_probabilities.index.values)
            self.assertAlmostEqual(
                expected_priors.loc[indel, "freq"],
                indel_probabilities.loc[indel, "freq"],
                delta=0.01,
            )

    def test_compute_indel_priors_with_conflicts(self):
        indel_probabilities = cas.pp.compute_indel_priors(self.alleletable_conflict, per_site=False)

        expected_priors = pd.DataFrame.from_dict(
            {
                "ATC": [2, 2 / 3],
                "GGG": [1, 1 / 3],
                "GAA": [1, 1 / 3],
                "AAA": [1, 1 / 3],
                "ACT": [1, 1 / 3],
                "TTT": [1, 1 / 3],
                "ATA": [2, 2 / 3],
            },
            orient="index",
            columns=["count", "freq"],
        )

        for indel in expected_priors.index:
            self.assertIn(indel, indel_probabilities.index.values)
            self.assertAlmostEqual(
                expected_priors.loc[indel, "freq"],
                indel_probabilities.loc[indel, "freq"],
                delta=0.01,
            )

    def test_compute_indel_priors_multiple_variables(self):
        indel_probabilities = cas.pp.compute_indel_priors(
            self.allele_table_mouse, grouping_variables=["Mouse", "intBC"], per_site=False
        )

        expected_priors = pd.DataFrame.from_dict(
            {
                "AAB": [2, 2 / 5],
                "BAB": [2, 2 / 5],
                "CAB": [2, 2 / 5],
                "AAA": [2, 2 / 5],
                "BAA": [2, 2 / 5],
                "CAA": [2, 2 / 5],
                "AAC": [1, 1 / 5],
                "BAC": [1, 1 / 5],
                "CAC": [1, 1 / 5],
                "AAD": [1, 1 / 5],
                "BAD": [1, 1 / 5],
                "CAD": [1, 1 / 5],
                "ABA": [1, 1 / 5],
                "BBA": [1, 1 / 5],
                "CBA": [1, 1 / 5],
                "ABB": [1, 1 / 5],
                "BBB": [1, 1 / 5],
                "CBB": [1, 1 / 5],
            },
            orient="index",
            columns=["count", "freq"],
        )

        for indel in expected_priors.index:
            self.assertIn(indel, indel_probabilities.index.values)
            self.assertAlmostEqual(
                expected_priors.loc[indel, "freq"],
                indel_probabilities.loc[indel, "freq"],
                delta=0.01,
            )

        # make sure permuting grouping variables doesn't change result
        indel_probabilities = cas.pp.compute_indel_priors(
            self.allele_table_mouse, grouping_variables=["intBC", "Mouse"], per_site=False
        )

        for indel in expected_priors.index:
            self.assertIn(indel, indel_probabilities.index.values)
            self.assertAlmostEqual(
                expected_priors.loc[indel, "freq"],
                indel_probabilities.loc[indel, "freq"],
                delta=0.01,
            )

    def test_compute_indel_priors_multiple_variables_noncassiopeia_alleletable(
        self,
    ):
        noncassiopeia_at = self.allele_table_mouse.copy()
        noncassiopeia_at.rename(columns={"r1": "cs1", "r2": "cs2", "r3": "cs3"}, inplace=True)
        indel_probabilities = cas.pp.compute_indel_priors(
            noncassiopeia_at,
            grouping_variables=["Mouse", "intBC"],
            cut_sites=["cs1", "cs2", "cs3"],
            per_site=False,
        )

        expected_priors = pd.DataFrame.from_dict(
            {
                "AAB": [2, 2 / 5],
                "BAB": [2, 2 / 5],
                "CAB": [2, 2 / 5],
                "AAA": [2, 2 / 5],
                "BAA": [2, 2 / 5],
                "CAA": [2, 2 / 5],
                "AAC": [1, 1 / 5],
                "BAC": [1, 1 / 5],
                "CAC": [1, 1 / 5],
                "AAD": [1, 1 / 5],
                "BAD": [1, 1 / 5],
                "CAD": [1, 1 / 5],
                "ABA": [1, 1 / 5],
                "BBA": [1, 1 / 5],
                "CBA": [1, 1 / 5],
                "ABB": [1, 1 / 5],
                "BBB": [1, 1 / 5],
                "CBB": [1, 1 / 5],
            },
            orient="index",
            columns=["count", "freq"],
        )

        for indel in expected_priors.index:
            self.assertIn(indel, indel_probabilities.index.values)
            self.assertAlmostEqual(
                expected_priors.loc[indel, "freq"],
                indel_probabilities.loc[indel, "freq"],
                delta=0.01,
            )

        # make sure permuting grouping variables doesn't change result
        indel_probabilities = cas.pp.compute_indel_priors(
            self.allele_table_mouse, grouping_variables=["intBC", "Mouse"], per_site=False
        )

        for indel in expected_priors.index:
            self.assertIn(indel, indel_probabilities.index.values)
            self.assertAlmostEqual(
                expected_priors.loc[indel, "freq"],
                indel_probabilities.loc[indel, "freq"],
                delta=0.01,
            )


if __name__ == "__main__":
    unittest.main()
