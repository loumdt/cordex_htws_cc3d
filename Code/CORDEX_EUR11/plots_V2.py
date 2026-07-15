from dataclasses import dataclass
import pandas as pd
from typing import List, Optional, Dict
from scipy import stats
import numpy as np
from highlight_text import ax_text
from abc import ABC

import matplotlib.pyplot as plt



@dataclass
class ColorScheme:
    """Class to manage color schemes for plots."""

    black = "#0D160B"
    grey = "#7D7D7D"
    periods = ["#9CAFB7","#7D7D7D",
     "#E8AE68", "#FA9500", "#B22222","#9D1E1E"]

    def get_color(self, period: int) -> str:
        return self.periods[period] if 0 <= period < len(self.periods) else self.grey


# Dictionary to manage units for plots
UNITS = {
    "Intensity": "°C",
    "Spatial extent": "km²",
    "Duration": "Days",
    "HWMId_pop_ssp1_all_period": "IQR-persons",
    "HWMId_pop": "IQR-persons",
}


class BasePlotter(ABC):
    """Abstract base class for plotters."""

    DEFAULT_CONFIG = {
        "dpi": 200,
        "font_size": 12,
    }

    def __init__(
        self,
        ax: Optional[plt.Axes] = None,
        language: str = "en",
    ):
        # FIGURE CONFIGURATION
        self.ax = ax or plt.gca()
        self.fig = self.ax.figure
        self._config = {}
        self.update_config(**self.DEFAULT_CONFIG)

        # LANGUAGE
        self.language = language
        self._get_localized_text = Translator(language=language).get_localized_text

    def plot(self, *args, **kwargs):
        raise NotImplementedError("Subclasses should implement this method.")

    def save(self, filepath: str):
        """Save the current figure to a file."""
        self.fig.savefig(
            filepath, dpi=self._config["dpi"], bbox_inches="tight", transparent=True
        )

    def update_config(self, **kwargs):
        self._config.update(kwargs)
        self.fig.set_dpi(self._config["dpi"])
        plt.rcParams.update({"font.size": self._config["font_size"]})


class PeriodDistributionPlotter(BasePlotter):
    """Class to plot the distribution of a variable over specified periods.

    This class creates distribution plots (KDE or histogram) showing the distribution
    of a variable across different time periods, with optional reference events.

    Args:
        ax (Optional[plt.Axes]): Matplotlib Axes object to plot on. If None, uses current Axes.
        language (str): Language code for translations (default is "en").
        colors (Optional[ColorScheme]): Color scheme to use for the plot. If None, uses default colors.
        units (dict): Dictionary mapping variable names to their units for labeling.

    Configuration Parameters:
        All visual aspects of the plot can be customized using the update_config() method.
        Available parameters include:

        Display Mode:
            - display_mode (str, default="kde"): Type of distribution plot. Either "kde"
              for Kernel Density Estimation, "histogram" for bar histograms, or
              "cumulative_histogram" for a decreasing cumulative histogram.

        KDE Configuration:
            - kde_resolution (int, default=1000): Number of points for KDE evaluation. Higher values
              create smoother curves but increase computation time.
            - kde_height_scale (float, default=1.5): Scale factor for distribution height
              normalization. Controls how tall the curves/bars appear relative to the period
              spacing. Also used by histograms.
            - kde_alpha (float, default=0.99): Transparency of distribution fills
              (0=transparent, 1=opaque). Also used by histograms.
            - kde_edge_linewidth (float, default=0.5): Line width for distribution edges.
              Also used by histograms.

        Histogram Configuration:
            - hist_bins (int, default=20): Number of bins for histogram mode.
            - hist_rwidth (float, default=0.9): Relative width of histogram bars (0-1).

        Reference Events Configuration:
            - reference_event_linewidth (float, default=2.5): Line width for vertical reference lines.
            - reference_event_label_fontsize (str, default="large"): Font size for reference event labels.
            - reference_event_label_rotation (float, default=45): Rotation angle in degrees for labels.
            - reference_event_label_bbox_linewidth (float, default=2): Border width for label boxes.

        Period Configuration:
            - period_line_width (float, default=0.5): Line width for horizontal period separator lines.
            - period_label_fontsize (str, default="small"): Font size for period labels.

        Score Configuration:
            - score_offset (float, default=0.6): Vertical offset for percentage score labels from
              the period baseline.
            - score_fontsize (str, default="small"): Font size for percentage scores.
            - score_bbox_linewidth (float, default=0.67): Border width for score label boxes.
            - score_bbox_pad (float, default=0.2): Padding inside score label boxes.
            - show_first_period_scores (bool, default=True): Whether to display percentage scores
              for the first (historical) period.

        Tick Configuration:
            - tick_length_x (float, default=2): Length of x-axis tick marks.
            - tick_length_y (float, default=0): Length of y-axis tick marks.
            - tick_pad_x (float, default=2): Padding between x-axis ticks and labels.
            - tick_pad_y (float, default=10): Padding between y-axis ticks and labels.
            - tick_labelsize_x (str, default="small"): Font size for x-axis tick labels.
            - tick_labelsize_y (str, default="medium"): Font size for y-axis tick labels.

        Label Configuration:
            - xlabel_fontsize (str, default="small"): Font size for x-axis label.
            - xlabel_fontstyle (str, default="italic"): Font style for x-axis label.

    Example:
        >>> plotter = PeriodDistributionPlotter()
        >>> plotter.update_config(display_mode="histogram", hist_bins=15)
        >>> plotter.plot(data, "Intensity", periods_dict, reference_events)
    """

    def __init__(
        self,
        ax: Optional[plt.Axes] = None,
        language: str = "en",
        colors: Optional[ColorScheme] = None,
        units: Optional[dict] = None,
    ):
        # FIGURE CONFIGURATION
        super().__init__(ax=ax, language=language)
        self.update_config(
            # Display Mode
            display_mode="kde",  # "kde", "histogram", or "cumulative_histogram"
            # KDE Configuration
            kde_resolution=1000,  # Number of points for KDE evaluation
            kde_height_scale=1.5,  # Scale factor for distribution height normalization
            kde_alpha=0.99,  # Transparency of distribution fills
            kde_edge_linewidth=0.5,  # Line width for distribution edges
            # Histogram Configuration
            hist_bins=20,  # Number of bins for histogram mode
            hist_rwidth=0.9,  # Relative width of histogram bars
            # Reference Events Configuration
            reference_event_linewidth=2.5,  # Line width for reference event vertical lines
            reference_event_label_fontsize="large",  # Font size for reference event labels
            reference_event_label_rotation=45,  # Rotation angle for reference event labels
            reference_event_label_bbox_linewidth=2,  # Border width for reference event label boxes
            # Period Configuration
            period_line_width=0.5,  # Line width for horizontal period separator lines
            period_label_fontsize="small",  # Font size for period labels
            # Score Configuration
            score_offset=0.6,  # Vertical offset for percentage score labels
            score_fontsize="small",  # Font size for percentage scores
            score_bbox_linewidth=0.67,  # Border width for score label boxes
            score_bbox_pad=0.2,  # Padding inside score label boxes
            show_first_period_scores=True,  # Whether to display percentage scores for the first period
            # Tick Configuration
            tick_length_x=2,  # Length of x-axis ticks
            tick_length_y=0,  # Length of y-axis ticks
            tick_pad_x=2,  # Padding between x-axis ticks and labels
            tick_pad_y=10,  # Padding between y-axis ticks and labels
            tick_labelsize_x="small",  # Font size for x-axis tick labels
            tick_labelsize_y="medium",  # Font size for y-axis tick labels
            # Label Configuration
            xlabel_fontsize="small",  # Font size for x-axis label
            xlabel_fontstyle="italic",  # Font style for x-axis label
        )
        # COLORS
        if colors is not None:
            self.colors = colors
        else:
            self.colors = ColorScheme()

        # UNITS
        if units is None:
            self._units = UNITS
        else:
            self._units = units

        self._highlighted_periods = dict()

    def get_configurable_parameters(self) -> Dict[str, str]:
        """Get a dictionary of all configurable parameters with their descriptions.

        Returns:
            Dict[str, str]: Dictionary mapping parameter names to their descriptions.
        """
        return {
            # Display Mode
            "display_mode": "Type of distribution plot: 'kde', 'histogram', or 'cumulative_histogram'",
            # KDE Configuration
            "kde_resolution": "Number of points for KDE evaluation (higher = smoother curves)",
            "kde_height_scale": "Scale factor for distribution height normalization (used by both KDE and histogram)",
            "kde_alpha": "Transparency of distribution fills (used by both KDE and histogram)",
            "kde_edge_linewidth": "Line width for distribution edges (used by both KDE and histogram)",
            # Histogram Configuration
            "hist_bins": "Number of bins for histogram mode",
            "hist_rwidth": "Relative width of histogram bars (0-1)",
            # Reference Events Configuration
            "reference_event_linewidth": "Line width for vertical reference lines",
            "reference_event_label_fontsize": "Font size for reference event labels",
            "reference_event_label_rotation": "Rotation angle in degrees for labels",
            "reference_event_label_bbox_linewidth": "Border width for label boxes",
            # Period Configuration
            "period_line_width": "Line width for horizontal period separator lines",
            "period_label_fontsize": "Font size for period labels",
            # Score Configuration
            "score_offset": "Vertical offset for percentage score labels",
            "score_fontsize": "Font size for percentage scores",
            "score_bbox_linewidth": "Border width for score label boxes",
            "score_bbox_pad": "Padding inside score label boxes",
            "show_first_period_scores": "Whether to display percentage scores for the first period",
            # Tick Configuration
            "tick_length_x": "Length of x-axis tick marks",
            "tick_length_y": "Length of y-axis tick marks",
            "tick_pad_x": "Padding between x-axis ticks and labels",
            "tick_pad_y": "Padding between y-axis ticks and labels",
            "tick_labelsize_x": "Font size for x-axis tick labels",
            "tick_labelsize_y": "Font size for y-axis tick labels",
            # Label Configuration
            "xlabel_fontsize": "Font size for x-axis label",
            "xlabel_fontstyle": "Font style for x-axis label",
        }

    def _check_dataframe(self, df: pd.DataFrame, columns: List[str]):
        return all(col in df.columns for col in columns)

    def _hide_spines(self):
        """Cache les bordures inutiles du graphique."""
        for spine in ["left", "right", "top"]:
            self.ax.spines[spine].set_visible(False)

    def _configure_tick_params(self):
        """Configure tick parameters for both axes."""
        self.ax.tick_params(
            axis="x",
            bottom=True,
            top=False,
            labelbottom=True,
            labeltop=False,
            length=self._config["tick_length_x"],
            pad=self._config["tick_pad_x"],
            labelsize=self._config["tick_labelsize_x"],
            labelcolor=self.colors.black,
            labelrotation=0,
            gridOn=False,
        )
        self.ax.tick_params(
            axis="y",
            left=True,
            right=False,
            labelleft=True,
            labelright=False,
            length=self._config["tick_length_y"],
            pad=self._config["tick_pad_y"],
            labelsize=self._config["tick_labelsize_y"],
            labelcolor=self.colors.black,
            gridOn=False,
        )

    def _calculate_value_range(self, data, variable, min_value, max_value):
        """Calculate min/max values for the variable if not provided."""
        _mx = (
            max_value
            if max_value is not None and np.isfinite(max_value)
            else data[variable].max()
        )
        _mn = (
            min_value
            if min_value is not None and np.isfinite(min_value)
            else data[variable].min()
        )
        return _mn, _mx

    def _add_event_label(self, reference_event, variable):
        """Add formatted text label for a reference event."""
        label_text = self._get_localized_text(reference_event[1]["label"])

        self.ax.text(
            x=reference_event[1][variable],
            y=len(self._highlighted_periods),
            s=label_text,
            ha="left",
            va="bottom",
            fontsize=self._config["reference_event_label_fontsize"],
            fontweight="bold",
            color="white",
            rotation=self._config["reference_event_label_rotation"],
            zorder=3.1,
            bbox=dict(
                facecolor=self.colors.black,
                edgecolor=self.colors.black,
                alpha=1,
                linewidth=self._config["reference_event_label_bbox_linewidth"],
                boxstyle="round",
            ),
            rotation_mode="anchor",
        )
    
    def _add_event_label_other(self, reference_event, variable):
        """Add formatted text label for a reference event."""
        label_text = self._get_localized_text(reference_event[1]["label"])

        self.ax.text(
            x=reference_event[1][variable],
            y=len(self._highlighted_periods),
            s=label_text,
            ha="left",
            va="bottom",
            fontsize=self._config["reference_event_label_fontsize"],
            fontweight="bold",
            color="white",
            rotation=self._config["reference_event_label_rotation"],
            zorder=3.1,
            bbox=dict(
                facecolor='red',
                edgecolor='red',
                alpha=1,
                linewidth=self._config["reference_event_label_bbox_linewidth"],
                boxstyle="round",
            ),
            rotation_mode="anchor",
        )

    def _merge_overlapping_events(self, reference_events, variable):
        """Merge reference events that share the same value for a variable.

        When multiple events have the same value, their labels are combined
        with ' / ' separator. Returns a DataFrame with unique values and
        merged labels.
        """

        return (
            reference_events.groupby(variable, sort=False)
            .agg(label=("label", " / ".join))
            .reset_index()
        )

    def _plot_reference_events(self, reference_events, variable):
        """Draw vertical lines and labels for reference events."""
        
        # Text labels with boxes
        for i,reference_event in enumerate(reference_events.iterrows()):
            if i==3:
                # Vertical lines
                self.ax.vlines(
                    reference_event[1][variable],
                    ymin=0,
                    ymax=len(self._highlighted_periods),
                    color='red',
                    linestyle="--",
                    linewidth=self._config["reference_event_linewidth"],
                    zorder=3,
                )
                self._add_event_label_other(reference_event, variable)
            else:
                # Vertical lines
                self.ax.vlines(
                    reference_event[1][variable],
                    ymin=0,
                    ymax=len(self._highlighted_periods),
                    color=self.colors.black,
                    linestyle="--",
                    linewidth=self._config["reference_event_linewidth"],
                    zorder=3,
                )
                self._add_event_label(reference_event, variable)

    def _calculate_percentile_scores(self, data, reference_events, variable):
        """Calculate percentile scores for reference events across periods."""
        # 1. Sort thresholds to ensure monotonic bins
        sorted_thresholds = reference_events.set_index("label")[variable].sort_values()
        labels = list(sorted_thresholds.index)

        # 2. Define bin edges: [-inf, T1, T2, ..., TN, inf]
        bins = np.concatenate(([-np.inf], sorted_thresholds.values, [np.inf]))

        # 3. Create labels for the N+1 bins
        # The first N bins take the threshold labels, the last takes label_max
        out_labels = labels + [f"_max"]

        # 4. Segment the data and calculate frequencies
        # include_lowest=True ensures the interval is [min, T1] for the first bin
        distribution = pd.concat(
            [
                pd.cut(
                    data.query(f"`{period}`")[variable],
                    bins=bins,
                    labels=out_labels,
                    include_lowest=True,
                )
                .value_counts(normalize=True)
                .reindex(out_labels)  # Ensure order and handle empty bins
                .fillna(0)
                .to_frame(name=period)
                for period in self._highlighted_periods.keys()
            ],
            axis=1,
        )

        return distribution * 100

    def _calculate_cumulative_scores(self, data, reference_events, variable):
        """Calculate cumulative scores (>= threshold) for reference events."""
        thresholds, labels = self._get_cumulative_thresholds(
            reference_events, variable, None, None
        )
        return self._calculate_cumulative_scores_for_thresholds(
            data, variable, thresholds, labels
        )

    def _get_cumulative_thresholds(
        self, reference_events, variable, min_value=None, max_value=None
    ):
        """Get sorted thresholds for cumulative scores within bounds."""
        sorted_thresholds = reference_events.set_index("label")[variable].sort_values()
        thresholds = sorted_thresholds.to_numpy()
        labels = list(sorted_thresholds.index)

        if (
            min_value is not None
            and max_value is not None
            and np.isfinite(min_value)
            and np.isfinite(max_value)
            and min_value > max_value
        ):
            min_value, max_value = max_value, min_value

        if min_value is not None and np.isfinite(min_value):
            keep = thresholds >= min_value
            thresholds = thresholds[keep]
            labels = [labels[i] for i in np.where(keep)[0]]

            if thresholds.size == 0 or not np.isclose(thresholds[0], min_value):
                thresholds = np.concatenate(([min_value], thresholds))
                labels = ["_min"] + labels

        if max_value is not None and np.isfinite(max_value):
            keep = thresholds <= max_value
            thresholds = thresholds[keep]
            labels = [labels[i] for i in np.where(keep)[0]]

        return thresholds, labels

    def _calculate_cumulative_scores_for_thresholds(
        self, data, variable, thresholds, labels
    ):
        """Calculate cumulative scores (>= threshold) for given thresholds."""
        scores = {}
        for period in self._highlighted_periods.keys():
            values = data.query(f"`{period}`")[variable].dropna().to_numpy()
            if values.size == 0:
                scores[period] = np.zeros_like(thresholds, dtype=float)
                continue
            scores[period] = (
                (values[:, None] >= thresholds[None, :]).mean(axis=0) * 100
            )

        return pd.DataFrame(scores, index=labels)

    def _get_distribution_plot_kwargs(self, period_index,color):
        """Get common plot kwargs for both KDE and histogram."""
        return dict(
            color=color,#self.colors.get_color(period_index),
            zorder=1
            + (len(self._highlighted_periods) - 1 - period_index)
            / len(self._highlighted_periods),
            alpha=self._config["kde_alpha"],
            linewidth=self._config["kde_edge_linewidth"],
            edgecolor=self.colors.black,
        )

    def _plot_single_kde(
        self, events, variable, period_index, my_color, max_value=None, min_value=None,
    ):
        """Plot KDE for a single period."""
        _kde = stats.gaussian_kde(events[variable].values)
        x_values = np.linspace(
            events[variable].min() if min_value is None else min_value,
            events[variable].max() if max_value is None else max_value,
            self._config["kde_resolution"],
        )
        kde_values = _kde(x_values)
        kde_values = (
            kde_values / kde_values.max() * self._config["kde_height_scale"]
        ) + period_index

        self.ax.fill_between(
            x_values,
            [period_index] * len(kde_values),
            kde_values,
            **self._get_distribution_plot_kwargs(period_index,my_color),
        )

    def _plot_single_histogram(
        self, events, variable, period_index, my_color, max_value=None, min_value=None
    ):
        """Plot histogram for a single period."""
        values = events[variable].values
        _mn = events[variable].min() if min_value is None else min_value
        _mx = events[variable].max() if max_value is None else max_value
        bin_edges = np.linspace(_mn, _mx, self._config["hist_bins"] + 1)

        counts, _ = np.histogram(values, bins=bin_edges)
        # Normalize so tallest bar matches kde_height_scale
        bar_heights = (
            counts / counts.max() * self._config["kde_height_scale"]
            if counts.max() > 0
            else counts
        )

        plot_kwargs = self._get_distribution_plot_kwargs(period_index,my_color)

        self.ax.bar(
            x=(bin_edges[:-1] + bin_edges[1:]) / 2,
            height=bar_heights,
            width=np.diff(bin_edges) * self._config["hist_rwidth"],
            bottom=period_index,
            align="center",
            **plot_kwargs,
        )

    def _plot_single_cumulative_histogram(
        self, events, variable, period_index, my_color, max_value=None, min_value=None,
    ):
        """Plot cumulative histogram for a single period (from 100% to 0%)."""
        values = events[variable].dropna().to_numpy()
        if values.size == 0:
            return

        unique_vals, counts = np.unique(values, return_counts=True)
        total = counts.sum()
        height_scale = self._config["kde_height_scale"]

        bound_min = unique_vals[0] if min_value is None else min_value
        bound_max = unique_vals[-1] if max_value is None else max_value
        if not np.isfinite(bound_min) or not np.isfinite(bound_max):
            return
        if bound_min > bound_max:
            bound_min, bound_max = bound_max, bound_min

        cum_counts = np.cumsum(counts)
        in_bounds = (unique_vals >= bound_min) & (unique_vals <= bound_max)
        x_points = unique_vals[in_bounds]

        def _survival_at(x, side="left"):
            if side == "right":
                idx = np.searchsorted(unique_vals, x, side="right")
            else:
                idx = np.searchsorted(unique_vals, x, side="left")
            count_le = cum_counts[idx - 1] if idx > 0 else 0
            return (total - count_le) / total * height_scale

        x_vals = [bound_min]
        y_vals = [_survival_at(bound_min, side="left")]
        current_x = bound_min
        current_y = y_vals[0]

        for x in x_points:
            if x != current_x:
                x_vals.append(x)
                y_vals.append(current_y)
            current_y = _survival_at(x, side="right")
            x_vals.append(x)
            y_vals.append(current_y)
            current_x = x

        if bound_max != current_x:
            x_vals.append(bound_max)
            y_vals.append(_survival_at(bound_max, side="left"))

        plot_kwargs = self._get_distribution_plot_kwargs(period_index,my_color)
        self.ax.fill_between(
            x_vals,
            [period_index] * len(x_vals),
            period_index + np.array(y_vals),
            **plot_kwargs,
        )

    def _add_percentage_scores(
        self,
        scores,
        highlighted_period,
        scores_positions,
        period_index,
        score_offset,
    ):
        """Add percentage score labels to the plot."""
        for x_pos, score in zip(scores_positions, scores[highlighted_period].values):
            ax_text(
                x=x_pos,
                y=period_index + score_offset,
                s=f"{np.abs(score):.0f}%",  # Absolute value to avoid negative zero
                ha="center",
                va="center",
                fontsize=self._config["score_fontsize"],
                fontstyle="italic",
                fontweight="bold",
                ax=self.ax,
                bbox=dict(
                    facecolor="whitesmoke",
                    alpha=1,
                    linewidth=self._config["score_bbox_linewidth"],
                    boxstyle=f"round,pad={self._config['score_bbox_pad']}",
                ),
                zorder=4,
            )

    def _add_cumulative_score_lines(
        self,
        scores,
        highlighted_period,
        score_thresholds,
        period_index,
        min_value,
        max_value,
    ):
        """Add cumulative score lines and labels for a period."""
        if score_thresholds is None or len(score_thresholds) == 0:
            return

        if max_value is None or not np.isfinite(max_value):
            return

        if min_value is not None and np.isfinite(min_value):
            x_range = max_value - min_value
        else:
            x_range = max_value - np.min(score_thresholds)
        x_range = x_range if np.isfinite(x_range) and x_range > 0 else 0
        x_pad = x_range * 0.01

        n_lines = len(score_thresholds)
        # offsets = np.linspace(1.0 / (n_lines + 1), n_lines / (n_lines + 1), n_lines)[::-1]

        for (x_start, score) in zip(
            score_thresholds, scores[highlighted_period].values
        ):
            #if np.isclose(score, 100.0):
            #    continue
            if score>95:
                continue
            y_line = period_index + score / 100 * self._config["kde_height_scale"]
            # self.ax.hlines(
            #     y_line,
            #     x_start,
            #     max_value,
            #     color=self.colors.black,
            #     linewidth=self._config["kde_edge_linewidth"],
            #     zorder=3,
            # )
            ax_text(
                x=x_start + x_pad,
                y=y_line,
                s=f"{np.abs(score):.0f}%",
                ha="center",
                va="bottom" if score <= 50 else "top",
                fontsize=self._config["score_fontsize"],
                fontstyle="italic",
                fontweight="bold",
                ax=self.ax,
                bbox=dict(
                    facecolor="whitesmoke",
                    alpha=1,
                    linewidth=self._config["score_bbox_linewidth"],
                    boxstyle=f"round,pad={self._config['score_bbox_pad']}",
                ),
                zorder=4,
            )

    def _plot_distributions(
        self,
        data,
        variable,
        scores_reference_events,
        score_positions_reference_events,
        my_color,
        score_thresholds_reference_events=None,
        max_value=None,
        min_value=None,
    ):
        """Plot distributions (KDE or histogram) for each highlighted period."""
        if self._config["display_mode"] == "histogram":
            plot_fn = self._plot_single_histogram
        elif self._config["display_mode"] == "cumulative_histogram":
            plot_fn = self._plot_single_cumulative_histogram
        else:
            plot_fn = self._plot_single_kde
        for i, highlighted_period in enumerate(self._highlighted_periods.keys()):
            # Get events for this period
            _events = data.query(f"`{highlighted_period}`")

            # Plot distribution
            plot_fn(_events, variable, i, my_color=my_color, max_value=max_value, min_value=min_value)

            # Add percentage scores
            # Skip first period if show_first_period_scores is False
            if (
                scores_reference_events is not None
                and not scores_reference_events.empty
                and (i > 0 or self._config["show_first_period_scores"])
            ):
                if self._config["display_mode"] == "cumulative_histogram":
                    self._add_cumulative_score_lines(
                        scores_reference_events,
                        highlighted_period,
                        score_thresholds_reference_events,
                        i,
                        min_value,
                        max_value,
                    )
                else:
                    self._add_percentage_scores(
                        scores_reference_events,
                        highlighted_period,
                        score_positions_reference_events,
                        i,
                        self._config["score_offset"],
                    )

    def _calculate_score_positions(self, reference_events, variable, _mn, _mx, _scores):
        """Calculate positions for displaying percentile scores."""
        _scores_positions_div = np.array(
            [
                _mn
                + (reference_events[variable].min() - _mn)
                / reference_events[variable].count()
            ]
            + [
                reference_events.loc[
                    reference_events["label"] == label, variable
                ].values[0]
                for label in _scores.index
                if label != "_max"
            ]
            + [_mx]
        )
        return (_scores_positions_div[:-1] + _scores_positions_div[1:]) / 2

    def _calculate_cumulative_score_positions(self, reference_events, variable):
        """Calculate positions for cumulative scores (at lower thresholds)."""
        sorted_thresholds = reference_events.set_index("label")[variable].sort_values()
        return sorted_thresholds.to_numpy()

    def _add_period_lines(self, min_value):
        """Add horizontal lines and labels for each period."""
        for i, period in enumerate(self._highlighted_periods.keys()):
            self.ax.axhline(
                i,
                color=self.colors.black,
                linestyle="-",
                linewidth=self._config["period_line_width"],
                zorder=1,
            )

            period_label = self._get_localized_text(self._highlighted_periods[period])
            ax_text(
                x=min_value,
                y=i,
                s=period_label,
                ha="left",
                va="bottom",
                fontsize=self._config["period_label_fontsize"],
                fontweight="bold",
                color=self.colors.black,
                ax=self.ax,
            )

    def _configure_labels_and_ticks(self, variable):
        """Configure axis labels, ticks, and formatting."""
        # Set xlabel with unit
        _unit = self._units.get(variable, None)
        if _unit is not None:
            unit_text = self._get_localized_text(_unit)
            self.ax.set_xlabel(
                unit_text,
                fontsize=self._config["xlabel_fontsize"],
                fontstyle=self._config["xlabel_fontstyle"],
                color=self.colors.grey,
            )

        self.ax.set_ylabel("")
        self.ax.yaxis.set_ticks([])

        # Configure tick parameters
        self._configure_tick_params()

    def _apply_plot_styling(self, min_value, max_value, variable):
        """Apply consistent styling to the plot."""
        # Set limits
        self.ax.set_xlim(min_value, max_value)
        self.ax.set_ylim(0, None)

        # Hide spines
        self._hide_spines()

        # Add period lines and labels
        self._add_period_lines(min_value)

        # Set labels and ticks
        self._configure_labels_and_ticks(variable)

    def plot(
        self,
        data: pd.DataFrame,
        variable: str,
        periods_columns_labels: Dict[str, str],
        reference_events: Optional[pd.DataFrame] = None,
        bounds: Optional[List[float]] = None,
        cut_kdes: bool = True,
        my_color: str = None,
    ):
        """Plot the distribution of a variable over specified periods.

        Args:
            data (pd.DataFrame): DataFrame containing the data to plot.
            variable (str): The variable/column name to plot.
            periods_columns_labels (Dict[str, str]): Dictionary mapping period column names (boolean columns saying whether an event belongs to a period) to their labels.
            reference_events (Optional[pd.DataFrame]): Optional DataFrame for data of reference events (same structure as data).
            bounds (Optional[List[float]]): Optional bounds for the x-axis.
            cut_kdes (bool): Whether to cut KDEs at the min/max values of the data. For
                cumulative histograms, bounds are always applied.
        """
        periods = list(periods_columns_labels.keys())
        if not self._check_dataframe(data, [variable] + periods):
            raise ValueError("DataFrame is missing required columns.")

        if reference_events is not None and not self._check_dataframe(
            reference_events, [variable]
        ):
            raise ValueError("Reference DataFrame is missing required columns.")

        # 1. Calculate min/max values for the variable if not provided
        if bounds is not None and len(bounds) != 2:
            raise ValueError("Bounds must be a list of two values: [min, max].")
        min_value, max_value = bounds if bounds is not None else (None, None)
        min_value, max_value = self._calculate_value_range(
            data, variable, min_value, max_value
        )

        # 2. Store highlighted periods for later use
        self._highlighted_periods = periods_columns_labels

        # 3. Starting by clearing the axes
        self.ax.clear()

        # 4. Getting the scores and their positions for the reference data
        _score_thresholds = None
        if reference_events is not None and not reference_events.empty:
            # Merge events with identical values (avoids duplicate bin edges
            # and overlapping labels)
            _merged_events = self._merge_overlapping_events(reference_events, variable)

            # Plot reference events with lines and labels
            self._plot_reference_events(_merged_events, variable)

            # Compute scores and positions for reference data
            if self._config["display_mode"] == "cumulative_histogram":
                _score_thresholds, _score_labels = self._get_cumulative_thresholds(
                    _merged_events, variable, min_value, max_value
                )
                _scores = self._calculate_cumulative_scores_for_thresholds(
                    data, variable, _score_thresholds, _score_labels
                )
                _scores_positions = None
            else:
                _scores = self._calculate_percentile_scores(
                    data, _merged_events, variable
                )
                _scores_positions = self._calculate_score_positions(
                    _merged_events, variable, min_value, max_value, _scores
                )
        else:
            _scores = None
            _scores_positions = None

        # 5. Plot distributions and scores
        use_bounds = (
            self._config["display_mode"] == "cumulative_histogram" or not cut_kdes
        )
        self._plot_distributions(
            data,
            variable,
            _scores,
            _scores_positions,
            my_color,
            score_thresholds_reference_events=_score_thresholds,
            max_value=max_value if use_bounds else None,
            min_value=min_value if use_bounds else None,
        )

        # 6. Apply styling
        self._apply_plot_styling(min_value, max_value, variable)
        return

    def explanatory_step1_kde_only(
        self,
        data: pd.DataFrame,
        variable: str,
        periods_columns_labels: Dict[str, str],
        bounds: Optional[List[float]] = None,
        cut_kdes: bool = True,
        my_color: 'str'="#E8AE68",
    ):
        """Create 1 figure explaining the first step to read the plot: the KDEs. It plots only the KDE for the first period."""
        self.plot(
            data=data,
            variable=variable,
            periods_columns_labels={
                list(periods_columns_labels.keys())[0]: list(
                    periods_columns_labels.values()
                )[0]
            },
            reference_events=None,
            bounds=bounds,
            cut_kdes=cut_kdes,
            my_color=my_color

        )
        return

    def explanatory_step2_add_reference_events(
        self,
        data: pd.DataFrame,
        variable: str,
        periods_columns_labels: Dict[str, str],
        reference_events: Optional[pd.DataFrame] = None,
        bounds: Optional[List[float]] = None,
        cut_kdes: bool = True,
        my_color: str = "#E8AE68",
    ):
        """Create 1 figure explaining the first step to read the plot: the KDEs. It plots only the KDE for the first period."""
        self.plot(
            data=data,
            variable=variable,
            periods_columns_labels={
                list(periods_columns_labels.keys())[0]: list(
                    periods_columns_labels.values()
                )[0]
            },
            reference_events=reference_events,
            bounds=bounds,
            cut_kdes=cut_kdes,
            my_color=my_color,
        )
        return


class Translator:
    """Class to translate text from english to target language for plots."""

    dictionaries = {
        "fr": {
            "Reanalysis": "Réanalyse",
            "Historical": "Historique",
            "Intensity": "Intensité",
            "Surface Area": "Surface",
            "Duration": "Durée",
            "Standardized Index": "Indice Standardisé",
            "% of France": "% de la France",
            "Months": "Mois",
            "Peak Date": "Date de Pic",
        }
    }

    def __init__(self, language: str = "en"):
        self.language = language

    def get_localized_text(self, text: str) -> str:
        if self.language == "en":
            return text
        return self.dictionaries.get(self.language, {}).get(text, text)
