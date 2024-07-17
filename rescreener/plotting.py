# rescreener.plotting

from typing import TYPE_CHECKING, Optional, Tuple

import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns
from matplotlib.font_manager import FontProperties

if TYPE_CHECKING:
    from .analysis import BootstrapAnalysis


class BootstrapPlot:
    """
    Base class for creating bootstrap plots using seaborn and matplotlib.

    This class provides a foundation for creating various types of plots
    related to bootstrap analysis results.
    """

    def __init__(
        self,
        xlabel: str,
        ylabel: str,
        title: str,
        figsize: Tuple[int, int] = (10, 5),
        dpi: int = 150,
        font_family: str = "sans-serif",
        font_size: int = 6,
        height: Optional[int] = None,
        width: Optional[int] = None,
    ):
        """
        Initialize a BootstrapPlot object.

        Args:
            xlabel (str): Label for the x-axis.
            ylabel (str): Label for the y-axis.
            title (str): Title of the plot.
            figsize (Tuple[int, int], optional): Size of the figure. Defaults to (10, 5).
            dpi (int, optional): Dots per inch for the figure. Defaults to 150.
        """
        self.seaborn = None  # Must be overridden in child classes
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.title = title
        self.xtick_rotation = False
        self.extra = None
        self.extra_kwargs = None
        self.font_family = font_family
        self.font_size = font_size
        self.italicize_genes = False
        self.ylim = None

        h, w = figsize
        figsize = (
            height if height is not None else h,
            width if width is not None else w,
        )

        # plt args
        self.plt_kwargs = dict(
            figsize=figsize,
            dpi=dpi,
        )

        self.grid_kwargs = {}

    def plot_extra(self):
        """
        Plots an extra layer on the plot if the attribute is set
        """
        if self.extra is not None:
            self.extra(**self.extra_kwargs)

    def plot(
        self,
        show: bool = True,
        save: Optional[str] = None,
    ):
        """
        Create and display the plot.

        Args:
            show (bool, optional): Whether to display the plot. Defaults to True.
            save (Optional[str], optional): File path to save the plot. Defaults to None.
        """

        plt.figure(**self.plt_kwargs)
        # Set font to Arial and font size to 6
        plt.rcParams["font.family"] = self.font_family
        plt.rcParams["font.size"] = self.font_size

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

        if self.italicize_genes:
            ax = plt.gca()
            fontstyle = FontProperties()
            fontstyle.set_style("italic")

            # Get current tick locations and labels
            locations = ax.get_xticks()
            labels = [item.get_text() for item in ax.get_xticklabels()]

            # Clear current tick labels
            ax.set_xticks([])

            # Set new tick locations and labels with italic font
            ax.set_xticks(locations)
            ax.set_xticklabels(labels, fontproperties=fontstyle)

        if self.ylim is not None:
            plt.ylim(self.ylim)

        self.plot_extra()

        plt.tight_layout()

        if save is not None:
            plt.savefig(save)
        if show:
            plt.show()


class Violins(BootstrapPlot):
    """
    Class for creating violin plots to visualize bootstrap analysis results.

    This class extends BootstrapPlot to create violin plots showing the
    distribution of overlapping hits in bootstraps compared to a standard.
    """

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
        ylim: Tuple[float, float] = (0, 1),
        draw_median_list: bool = True,
        inner_kwargs: dict = {},
        grid_kwargs: dict = {},
        sns_kwargs: dict = {},
        pointplot_kwargs: dict = {},
        **kwargs,
    ):
        """
        Initialize a Violins plot object.

        Args:
            bsa (BootstrapAnalysis): The BootstrapAnalysis object containing the data.
            xlabel (str, optional): Label for the x-axis. Defaults to "Number of Mice in Treatment Group".
            ylabel (str, optional): Label for the y-axis. Defaults to "Fraction of Overlapping Hits".
            title (str, optional): Title of the plot. Defaults to "Fraction of Overlapping Hits in Bootstraps compared to Standard".
            color (str, optional): Color of the violin plot. Defaults to "salmon".
            linewidth (int, optional): Width of the violin plot outline. Defaults to 2.
            alpha (float, optional): Transparency of the violin plot. Defaults to 0.8.
            linestyle (str, optional): Style of the violin plot outline. Defaults to "-".
            fill (bool, optional): Whether to fill the violin plot. Defaults to False.
            draw_median_list (bool): Whether to draw a pointplot to link the medians. Defaults to True.
            inner_kwargs (dict, optional): Additional kwargs for inner plot elements. Defaults to {}.
            grid_kwargs (dict, optional): Additional kwargs for grid. Defaults to {}.
            sns_kwargs (dict, optional): Additional kwargs for seaborn plot. Defaults to {}.
            **kwargs: Additional kwargs to pass to BootstrapPlot.__init__().
        """
        super().__init__(xlabel, ylabel, title, **kwargs)

        self.seaborn = sns.violinplot
        self.ylim = ylim

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

        if draw_median_list:
            self.extra = sns.pointplot
            self.extra_kwargs = dict(
                data=bsa.overlaps.group_by("subset").agg(
                    pl.col("frac_overlapping").median()
                ),
                x="subset",
                y="frac_overlapping",
                color="darkred",
                markers="o",
                linestyles="-",
                linewidth=1,
            )
            self.extra_kwargs.update(pointplot_kwargs)

        # grid args
        self.grid_kwargs = dict(
            axis="y",
            color="black",
            linestyle="--",
            alpha=0.3,
        )
        self.grid_kwargs.update(grid_kwargs)


class Recovery(BootstrapPlot):
    """
    Class for creating bar plots to visualize gene significance recovery across bootstraps.

    This class extends BootstrapPlot to create bar plots showing the proportion
    of significant tests for each gene across bootstrap iterations.
    """

    def __init__(
        self,
        bsa: "BootstrapAnalysis",
        xlabel: str = "Gene",
        ylabel: str = "Proportion of Significant Tests",
        color: str = "darkcyan",
        sns_kwargs: dict = {},
        relabel_tss: bool = True,
        italicize_genes: bool = True,
        **kwargs,
    ):
        """
        Initialize a Recovery plot object.

        Args:
            bsa (BootstrapAnalysis): The BootstrapAnalysis object containing the data.
            xlabel (str, optional): Label for the x-axis. Defaults to "Gene".
            ylabel (str, optional): Label for the y-axis. Defaults to "Proportion of Significant Tests".
            color (str, optional): Color of the bars. Defaults to "darkcyan".
            sns_kwargs (dict, optional): Additional kwargs for seaborn plot. Defaults to {}.
        """
        super().__init__(
            xlabel,
            ylabel,
            title=f"Distribution of gene significance across bootstraps (n={bsa.total_tests})",
            **kwargs,
        )

        if relabel_tss:
            bsa.recovery = bsa.recovery.with_columns(
                pl.col("gene").str.replace("_P1", "").str.replace("_P2", "")
            )

        self.seaborn = sns.barplot
        self.sns_kwargs = dict(
            data=bsa.recovery,
            x="gene",
            y="frac_tests",
            color=color,
            **sns_kwargs,
        )
        self.xtick_rotation = True
        self.italicize_genes = italicize_genes
