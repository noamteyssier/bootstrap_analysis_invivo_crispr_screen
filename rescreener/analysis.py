# rescreener.analysis

import os
import polars as pl
from typing import Optional
from glob import glob
from tqdm import tqdm

from ._constants import FULL_DIR, SUBSET_DIR, FULL_NAME_PREFIX

class BootstrapAnalysis:
    """
    A class for analyzing bootstrap results from CRISPR screen data.

    This class handles loading and analyzing data from full and subset CRISPR screens,
    calculating overlaps between bootstraps and a standard set, and measuring hit recovery.
    """

    def __init__(
        self,
        directory: str,
        standard: Optional[str] = None,
        fdr: float = 0.1,
    ):
        """
        Initialize the BootstrapAnalysis object.

        Args:
            directory (str): Path to the directory containing full and subset analysis results.
            standard (Optional[str], optional): Path to a file containing standard hits. Defaults to None.
            fdr (float, optional): False Discovery Rate threshold for considering hits. Defaults to 0.1.
        """
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
        """
        Validate the provided directory structure.

        Args:
            directory (str): Path to the main directory.

        Returns:
            Tuple[str, str, str]: Validated paths for main, full, and subset directories.

        Raises:
            ValueError: If any of the expected directories do not exist.
        """
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
        """
        Load the standard hits dataset.

        Args:
            standard (Optional[str]): Path to a file containing standard hits.

        Returns:
            pl.DataFrame: DataFrame containing standard hits.
        """
        if standard is not None:
            filename = standard
        else:
            filename = os.path.join(
                self.full_dir, f"{FULL_NAME_PREFIX}.gene_results.tsv"
            )

        return BootstrapAnalysis._load_hits_dataframe(filename, self.fdr)

    def _load_bootstraps(self) -> pl.DataFrame:
        """
        Load all bootstrap results from the subset directory.

        Returns:
            pl.DataFrame: Concatenated DataFrame of all bootstrap results.
        """
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
        Calculate the overlap between the standard and each bootstrap.

        Returns:
            pl.DataFrame: DataFrame containing overlap information for each bootstrap.
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
        Calculate how often a hit in the standard is observed across all bootstraps.

        Returns:
            pl.DataFrame: DataFrame containing hit recovery information for each gene.
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
        """
        Load a hits dataframe from a file and filter based on FDR.

        Args:
            filename (str): Path to the file containing hit data.
            fdr (float): False Discovery Rate threshold for filtering hits.

        Returns:
            pl.DataFrame: Filtered DataFrame containing hits.
        """
        return pl.read_csv(filename, separator="\t").filter(pl.col("fdr") < fdr)