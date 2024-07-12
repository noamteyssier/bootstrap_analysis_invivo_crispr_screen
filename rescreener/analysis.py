# rescreener.analysis

import os
import polars as pl
from typing import Optional
from glob import glob
from tqdm import tqdm

from ._constants import FULL_DIR, SUBSET_DIR, FULL_NAME_PREFIX


class BootstrapAnalysis:
    def __init__(
        self,
        directory: str,
        standard: Optional[str] = None,
        fdr: float = 0.1,
    ):
        (self.directory, self.full_dir, self.subset_dir) = self._validate_directory(
            directory
        )
        self.fdr = fdr

        # Read the standard (either a secondary provided hits file or generated from the full directory)
        self.standard = self._load_standard(standard)

        # Read in all the bootstraps
        self.bootstraps = self._load_bootstraps()

        # Measure overlap between groups
        self.overlaps = self._measure_overlap()

        # Measures standard gene representation in bootstraps
        self.recovery = self._measure_hit_recovery()

        print("Analysis loaded.")

    def _validate_directory(self, directory):
        directory = os.path.abspath(directory)
        full_dir = os.path.join(directory, FULL_DIR)
        subset_dir = os.path.join(directory, SUBSET_DIR)
        if not os.path.exists(directory):
            raise ValueError(f"The directory {directory} does not exist")
        if not os.path.exists(full_dir):
            raise ValueError(f"The expected directory {full_dir} does not exist")
        if not os.path.exists(subset_dir):
            raise ValueError(f"The expected directory {subset_dir} does not exist")
        return (
            directory,
            full_dir,
            subset_dir,
        )

    def _load_standard(self, standard: Optional[str]) -> pl.DataFrame:
        if standard is not None:
            filename = standard
        else:
            filename = os.path.join(
                self.full_dir, f"{FULL_NAME_PREFIX}.gene_results.tsv"
            )

        return BootstrapAnalysis._load_hits_dataframe(filename, self.fdr)

    def _load_bootstraps(self) -> pl.DataFrame:
        hit_files = glob(
            os.path.join(
                self.subset_dir,
                "*.gene_results.tsv",
            )
        )
        cohort_names = [
            os.path.basename(x).split(".gene_results")[0] for x in hit_files
        ]
        iterable = tqdm(
            zip(hit_files, cohort_names),
            desc="Loading hits from subsets",
            total=len(hit_files),
        )
        return pl.concat(
            [
                BootstrapAnalysis._load_hits_dataframe(path, self.fdr).with_columns(
                    pl.lit(name).alias("cohort"),
                    pl.lit(int(name.split("_")[-1])).alias("replicate"),
                    pl.lit(int(name.split("_")[-2])).alias("subset"),
                )
                for path, name in iterable
            ]
        )

    def _measure_overlap(self) -> pl.DataFrame:
        """
        Calculates the overlap between the standard and each bootstrap as a number and a fraction
        """
        print("Measuring set overlaps...")
        in_set = self.standard.select("gene").to_series().unique()
        return (
            self.bootstraps.group_by(["cohort", "replicate", "subset"])
            .agg(pl.col("gene").is_in(in_set).sum().alias("num_overlapping"))
            .with_columns(
                (pl.col("num_overlapping") / in_set.len()).alias("frac_overlapping")
            )
            .sort(["subset", "replicate"])
        )
    
    def _measure_hit_recovery(self) -> pl.DataFrame:
        """
        Calculates how often a hit in the standard is observed across all bootstraps
        """
        print("Measuring hit recovery...")
        in_set = self.standard.select("gene").to_series().unique()
        self.total_tests = self.bootstraps.select("cohort").n_unique()
        return (
            self.bootstraps.filter(pl.col("gene").is_in(in_set))
            .group_by("gene")
            .agg(pl.col("cohort").len().alias("num_tests"))
            .with_columns((pl.col("num_tests") / self.total_tests).alias("frac_tests"))
            .sort("frac_tests")
        )

    @staticmethod
    def _load_hits_dataframe(
        filename: str,
        fdr: float,
    ) -> pl.DataFrame:
        return pl.read_csv(filename, separator="\t").filter(pl.col("fdr") < fdr)
