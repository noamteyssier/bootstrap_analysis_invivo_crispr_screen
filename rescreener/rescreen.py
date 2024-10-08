# rescreener.rescreen

import multiprocessing
import os
import shutil
from subprocess import PIPE, Popen
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from ._constants import FULL_DIR, FULL_NAME_PREFIX, SUBSET_DIR, SUBSET_NAME_PREFIX


class Rescreener:
    """
    A class for performing CRISPR screen analysis on full and subset data.

    This class handles the setup, validation, and execution of CRISPR screen
    analysis for both the full dataset and bootstrapped subsets.
    """

    def __init__(
        self,
        table_path: str,
        reference_libraries: List[str],
        test_libraries: Optional[List[str]] = None,
        exclude_samples: Optional[List[str]] = None,
        prefix: str = "bootstraps",
        aggregation_method: str = "geopagg",
        use_product: bool = False,
        min_base_mean: Optional[int] = None,
        overwrite: bool = False,
        with_replacement: bool = True,
        n_threads: int = -1,
    ):
        """
        Initialize the Rescreener object.

        Args:
            table_path (str): Path to the input count matrix file.
            reference_libraries (List[str]): List of reference library names.
            test_libraries (Optional[List[str]], optional): List of test library names. Defaults to None.
            exclude_samples (Optional[List[str]], optional): List of samples to exclude. Defaults to None.
            prefix (str, optional): Prefix for output directory. Defaults to "bootstraps".
            aggregation_method (str, optional): Aggregation method for the CRISPR screen analysis. Defaults to "geopagg".
            use_product (bool, optional): Whether to use the gene scores instead of p-values in crispr_screen. Defaults to False.
            min_base_mean (Optional[int], optional): Minimum base mean for filtering. Defaults to None.
            overwrite (bool, optional): Whether to overwrite existing output. Defaults to False.
            with_replacement (bool, optional): Whether to sample with replacement. Defaults to True.
        """
        self.table_path = table_path
        self._columns = self._fetch_columns()

        self.reference_libraries = reference_libraries
        if test_libraries is not None:
            self.test_libraries = test_libraries
        else:
            self.test_libraries = self._build_test_libraries(exclude_samples)
        self._validate_sample_names()

        self.prefix = prefix
        self.overwrite = overwrite
        self.aggregation_method = aggregation_method
        self.use_product = use_product
        self.min_base_mean = min_base_mean
        self.with_replacement = with_replacement
        self.n_threads = n_threads

        self._validate_crispr_screen()
        self._validate_aggregation_method()
        self._initialize_output_dir()

    def _fetch_columns(self):
        """
        Fetch and validate the columns from the input count matrix.

        Returns:
            List[str]: List of column names from the input file.

        Raises:
            ValueError: If the input file does not have the expected column structure.
        """
        # Only load in a little of the matrix to match headers
        columns = pd.read_csv(self.table_path, nrows=5, sep="\t").columns.to_list()

        if columns[:2] != ["Guide", "Gene"]:
            raise ValueError(
                "Count matrix does not match expected columns from `sgcount`, expecting first two columns to be [ `Guide`, `Gene` ]. Found: {}".format(
                    columns[:2]
                )
            )
        return columns

    def _build_test_libraries(self, exclude_samples: Optional[List[str]]) -> List[str]:
        """
        Build a list of test libraries, excluding specified samples and reference libraries.

        Args:
            exclude_samples (Optional[List[str]]): List of samples to exclude.

        Returns:
            List[str]: List of test library names.

        Raises:
            ValueError: If no treatment libraries are found after exclusions.
        """
        exclusion = set(exclude_samples) if exclude_samples is not None else set()
        [exclusion.add(n) for n in self.reference_libraries]
        [exclusion.add(n) for n in ["Guide", "Gene"]]
        inclusion = [n for n in self._columns if n not in exclusion]
        if len(inclusion) == 0:
            raise ValueError(
                "No treatment libraries found - either they were all excluded for being in the reference library or too many exclusions were provided"
            )
        return inclusion

    def _validate_sample_names(self):
        """
        Validate that all reference and test library names are present in the input file.

        Raises:
            ValueError: If any reference or test sample is missing from the input file.
        """
        for name in self.reference_libraries:
            if name not in self._columns[2:]:
                raise ValueError(
                    f"Reference sample `{name}` is missing from columns in provided matrix"
                )

        for name in self.test_libraries:
            if name not in self._columns[2:]:
                raise ValueError(
                    f"Treatment sample `{name}` is missing from columns in provided matrix"
                )

    def _validate_crispr_screen(self):
        """
        Validate that the `crispr_screen` command is available in the system PATH.

        Raises:
            RuntimeError: If `crispr_screen` is not found in the PATH.
        """
        if shutil.which("crispr_screen") is None:
            raise RuntimeError(
                "Unable to find `crispr_screen` in `$PATH` - you will need to install it. Refer to https://noamteyssier.github.io/crispr_screen/install.html for details."
            )

    def _validate_aggregation_method(self):
        """
        Validate the provided aggregation method.

        Raises:
            ValueError: If the aggregation method is not one of the supported options.
        """
        supported_methods = ["geopagg", "rra", "inc"]
        if self.aggregation_method not in supported_methods:
            raise ValueError(
                f"Aggregation method {self.aggregation_method} is not supported. Choose from {supported_methods}."
            )

    def _initialize_output_dir(self):
        """
        Initialize the output directory structure.

        This method creates the main output directory and subdirectories for full and subset analyses.

        Raises:
            ValueError: If no prefix is provided.
            FileExistsError: If the output directory already exists and overwrite is False.
        """
        if self.prefix is None:
            raise ValueError(
                "A prefix must be provided to initialize the output directory."
            )

        self._out_dir = os.path.abspath(self.prefix)
        self._full_dir = os.path.join(self._out_dir, FULL_DIR)
        self._subset_dir = os.path.join(self._out_dir, SUBSET_DIR)

        # Check if the directory exists
        if os.path.exists(self._out_dir):
            if self.overwrite:
                # Remove the existing directory and its contents
                shutil.rmtree(self._out_dir)
            else:
                raise FileExistsError(
                    f"The directory {self._out_dir} already exists. Use overwrite=True to replace it."
                )

        # Create the directories
        os.makedirs(self._full_dir)
        os.makedirs(self._subset_dir)

    def run_original(self):
        """
        Run CRISPR screen analysis on the full set of test samples.

        This method executes the `crispr_screen` command on the complete dataset.
        """
        print("Starting crispr_screen...")
        Rescreener._run_crispr_screen(
            self.table_path,
            os.path.join(self._full_dir, FULL_NAME_PREFIX),
            self.reference_libraries,
            self.test_libraries,
            aggregation_method=self.aggregation_method,
            use_product=self.use_product,
            min_base_mean=self.min_base_mean,
        )
        print("Done.")

    def run_bootstraps(
        self,
        step_value=1,
        num_reps=50,
        seed=42,
    ):
        """
        Run CRISPR screen analysis on bootstrapped subsets of the test samples.

        This method creates and analyzes multiple subsets of the test libraries,
        with varying sizes and repetitions.

        Args:
            step_value (int, optional): Step size for increasing subset size. Defaults to 1.
            num_reps (int, optional): Number of repetitions for each subset size. Defaults to 50.
            seed (int, optional): Random seed for reproducibility. Defaults to 42.
        """
        np.random.seed(seed)

        # Prepare arguments for parallel execution
        args_list = []
        for subset_size in np.arange(1, len(self.test_libraries), step_value):
            for rep_index in range(num_reps):
                treatment_subset = np.random.choice(
                    self.test_libraries, subset_size, replace=self.with_replacement
                ).tolist()
                name = f"{SUBSET_NAME_PREFIX}_{subset_size}_{rep_index}"
                args_list.append((name, treatment_subset))

        # Use multiprocessing to run the analyses in parallel
        total_threads = self.n_threads if self.n_threads > 0 else None
        with multiprocessing.Pool(processes=total_threads) as pool:
            list(
                tqdm(
                    pool.imap(self._run_single_bootstrap, args_list),
                    total=len(args_list),
                    desc="Running bootstraps",
                )
            )

    def _run_single_bootstrap(self, args):
        name, treatment_subset = args
        Rescreener._run_crispr_screen(
            self.table_path,
            os.path.join(self._subset_dir, name),
            self.reference_libraries,
            treatment_subset,
            aggregation_method=self.aggregation_method,
            use_product=self.use_product,
            min_base_mean=self.min_base_mean,
        )

    @staticmethod
    def _run_crispr_screen(
        table_path: str,
        output_prefix: str,
        reference_libraries: List[str],
        test_libraries: List[str],
        aggregation_method: str = "geopagg",
        use_product: bool = False,
        min_base_mean: Optional[int] = None,
    ) -> Tuple[bytes, bytes]:
        """
        Execute the CRISPR screen analysis command.

        This static method constructs and runs the `crispr_screen` command
        with the provided parameters.

        Args:
            table_path (str): Path to the input count matrix file.
            output_prefix (str): Prefix for output files.
            reference_libraries (List[str]): List of reference library names.
            test_libraries (List[str]): List of test library names.

        Returns:
            Tuple[bytes, bytes]: Stdout and stderr output from the command execution.
        """
        args = []
        args.append("crispr_screen")
        args.append("test")
        args.append("-i")
        args.append(table_path)
        args.append("-o")
        args.append(output_prefix)
        args.append("-c")
        args.extend(reference_libraries)
        args.append("-t")
        args.extend(test_libraries)
        args.append("-g")
        args.append(aggregation_method)
        args.append("-T")
        args.append("1")
        if use_product:
            args.append("--use-product")
        if min_base_mean is not None:
            args.append("--min-base-mean")
            args.append(str(min_base_mean))

        cmd = Popen(args, stdout=PIPE, stderr=PIPE)
        stdout, stderr = cmd.communicate()
        return (stdout, stderr)

    @staticmethod
    def _check_crispr_screen() -> Tuple[bytes, bytes]:
        """
        Check the version of the installed CRISPR screen tool.

        This static method runs the `crispr_screen --version` command to verify
        the installation and get version information.

        Returns:
            Tuple[bytes, bytes]: Stdout and stderr output from the command execution.
        """
        args = []
        args.append("crispr_screen")
        args.append("--version")

        cmd = Popen(args, stdout=PIPE, stderr=PIPE)
        stdout, stderr = cmd.communicate()
        return (stdout, stderr)
