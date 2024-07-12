# rescreener.plotting

import seaborn as sns
import matplotlib.pyplot as plt
from typing import TYPE_CHECKING, Tuple, Optional

if TYPE_CHECKING:
    from .analysis import BootstrapAnalysis

class BootstrapPlot:
    def __init__(
        self,
        xlabel: str,
        ylabel: str,
        title: str,
        figsize: Tuple[int, int] = (10, 5),
        dpi: int = 150,
    ):
        self.seaborn = None  # Must override
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.title = title
        self.xtick_rotation = False

        # plt args
        self.plt_kwargs = dict(
            figsize=figsize,
            dpi=dpi,
        )

        self.grid_kwargs = {}

    def plot(
            self,
            show: bool = True,
            save: Optional[str] = None,
        ):
        plt.figure(**self.plt_kwargs)
        self.seaborn(
            **self.sns_kwargs,
        )
        if len(self.grid_kwargs) > 0:
            plt.grid(**self.grid_kwargs)

        plt.xlabel(self.xlabel)
        plt.ylabel(self.ylabel)
        plt.title(self.title)
        
        if self.xtick_rotation:
            plt.xticks(rotation=90)

        plt.tight_layout()

        if save is not None:
            plt.savefig(save)
        if show:
            plt.show()

class Violins(BootstrapPlot):
    def __init__(
        self,
        bsa: "BootstrapAnalysis",
        xlabel: str = "Number of Mice in Treatment Group",
        ylabel: str = "Fraction of Overlapping Hits",
        title: str = "Fraction of Overlapping Hits in Bootstraps compared to Standard",
        color: str = "salmon",
        linewidth: int = 2,
        alpha: float = 0.8,
        linestyle: str = "-",
        fill: bool = False,
        inner_kwargs: dict = {},
        grid_kwargs: dict = {},
        sns_kwargs: dict = {},
        **kwargs,
    ):
        super().__init__(xlabel, ylabel, title, **kwargs)
        
        self.seaborn = sns.violinplot

        # inner args
        self.sns_inner_kwargs = dict(
            box_width=7,
            whis_width=1,
            color="0.1",
        )
        self.sns_inner_kwargs.update(inner_kwargs)

        self.sns_kwargs = dict(
            data=bsa.overlaps,
            x="subset",
            y="frac_overlapping",
            color=color,
            inner_kws=self.sns_inner_kwargs,
            linewidth=linewidth,
            alpha=alpha,
            linestyle=linestyle,
            fill=fill,
            **sns_kwargs,
        )

        # grid args
        self.grid_kwargs = dict(
            axis="y",
            color="black",
            linestyle="--",
            alpha=0.3,
        )
        self.grid_kwargs.update(grid_kwargs)

class Recovery(BootstrapPlot):
    def __init__(
        self,
        bsa: "BootstrapAnalysis",
        xlabel: str = "Gene",
        ylabel: str = "Proportion of Significant Tests",
        color: str = "darkcyan",
        sns_kwargs: dict = {},
    ):
        super().__init__(xlabel, ylabel, title=f"Distribution of gene significance across bootstraps (n={bsa.total_tests})")
        
        self.seaborn = sns.barplot
        self.sns_kwargs = dict(
            data=bsa.recovery,
            x="gene",
            y="frac_tests",
            color=color,
            **sns_kwargs,
        )
        self.xtick_rotation = True