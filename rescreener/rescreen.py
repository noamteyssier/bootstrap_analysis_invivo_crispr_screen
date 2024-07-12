# rescreener.rescreen

import os
import shutil
import numpy as np
import pandas as pd
from typing import List, Optional, Tuple
from subprocess import Popen, PIPE
from tqdm.auto import tqdm

from ._constants import FULL_DIR, SUBSET_DIR, FULL_NAME_PREFIX, SUBSET_NAME_PREFIX


class Rescreener:
    def __init__(
        self,
        table_path: str,
        reference_libraries: List[str],
        test_libraries: Optional[List[str]] = None,
        exclude_samples: Optional[List[str]] = None,
        prefix: str = "bootstraps",
        overwrite: bool = False,
    ):
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

        self._validate_crispr_screen()
        self._initialize_output_dir()

    def _fetch_columns(self):
        """
        returns the columns in the table and validates the expected structure
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
        Ensures that `crispr_screen` is in the $PATH and will error out otherwise
        """
        if shutil.which("crispr_screen") is None:
            raise RuntimeError(
                "Unable to find `crispr_screen` in `$PATH` - you will need to install it. Refer to https://noamteyssier.github.io/crispr_screen/install.html for details."
            )

    def _initialize_output_dir(self):
        """
        Initializes the output directory and overwrites an existing one if the overwrite flag is provided

        # regex
        ./{prefix}/{full,subsets}
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
        runs `crispr_screen` on the full set of test samples
        """
        print("Starting crispr_screen...")
        Rescreener._run_crispr_screen(
            self.table_path,
            os.path.join(self._full_dir, FULL_NAME_PREFIX),
            self.reference_libraries,
            self.test_libraries,
        )
        print("Done.")

    def run_bootstraps(
        self,
        step_value=1,
        num_reps=50,
        seed=42,
    ):
        """
        runs `crispr_screen` on all bootstraps
        """

        cohorts = {}

        # Iterate over subsets of incrementing size
        for subset_size in tqdm(
            np.arange(1, len(self.test_libraries), step_value),
            desc="Running subsets",
            position=1,
        ):
            # Iterate over each instantiated boostrap
            for rep_index in tqdm(
                np.arange(num_reps), desc="Running boostraps", position=0
            ):
                # Randomly sample n test libraries with replacement
                treatment_subset = np.random.choice(
                    self.test_libraries, subset_size, replace=True
                )

                # Save the subset
                name = "{}_{}_{}".format(
                    SUBSET_NAME_PREFIX,
                    subset_size,
                    rep_index,
                )
                cohorts[name] = treatment_subset

                # run the screen
                Rescreener._run_crispr_screen(
                    self.table_path,
                    os.path.join(self._subset_dir, name),
                    self.reference_libraries,
                    treatment_subset,
                )

    @staticmethod
    def _run_crispr_screen(
        table_path: str,
        output_prefix: str,
        reference_libraries: List[str],
        test_libraries: List[str],
    ) -> Tuple[bytes, bytes]:
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

        cmd = Popen(args, stdout=PIPE, stderr=PIPE)
        stdout, stderr = cmd.communicate()
        return (stdout, stderr)

    @staticmethod
    def _check_crispr_screen() -> Tuple[bytes, bytes]:
        args = []
        args.append("crispr_screen")
        args.append("--version")

        cmd = Popen(args, stdout=PIPE, stderr=PIPE)
        stdout, stderr = cmd.communicate()
        return (stdout, stderr)
