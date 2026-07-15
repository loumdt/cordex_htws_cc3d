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
    periods = ["#9CAFB7", "#E8AE68", "#FA9500", "#B22222"]

    def get_color(self, period: int) -> str:
        return self.periods[period] if 0 <= period < len(self.periods) else self.grey


# Dictionary to manage units for plots
UNITS = {
    "Intensity": "°C",
    "Surface Area": "km²",
    "Duration": "Days",
    "Max": "°C",
    "HWMId_sum": "IQR",
    "Exposed_population_ghs": "people",
    "HWMId_pop_ghs": "IQR-person/km²",
    "Exposed_population_ssp1": "people",
    "HWMId_pop_ssp1": "IQR-person/km²",
    "Exposed_population_ssp2": "people",
    "HWMId_pop_ssp2": "IQR-person/km²",
    "Exposed_population_ssp3": "people",
    "HWMId_pop_ssp3": "IQR-person/km²",
    "Exposed_population_ssp4": "people",
    "HWMId_pop_ssp4": "IQR-person/km²",
    "Exposed_population_ssp5": "people",
    "HWMId_pop_ssp5": "IQR-person/km²",
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

    This class creates KDE (Kernel Density Estimation) plots showing the distribution
    of a variable across different time periods, with optional reference events.

    Args:
        ax (Optional[plt.Axes]): Matplotlib Axes object to plot on. If None, uses current Axes.
        language (str): Language code for translations (default is "en").
        colors (Optional[ColorScheme]): Color scheme to use for the plot. If None, uses default colors.
        units (dict): Dictionary mapping variable names to their units for labeling.

    Configuration Parameters:
        All visual aspects of the plot can be customized using the update_config() method.
        Available parameters include:

        KDE Configuration:
            - kde_resolution (int, default=1000): Number of points for KDE evaluation. Higher values
              create smoother curves but increase computation time.
            - kde_height_scale (float, default=1.5): Scale factor for KDE height normalization.
              Controls how tall the KDE curves appear relative to the period spacing.
            - kde_alpha (float, default=0.99): Transparency of KDE fills (0=transparent, 1=opaque).
            - kde_edge_linewidth (float, default=0.5): Line width for KDE curve edges.

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
        >>> plotter.update_config(kde_height_scale=2.0, score_fontsize="medium")
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
            # KDE Configuration
            kde_resolution=1000,  # Number of points for KDE evaluation
            kde_height_scale=1.5,  # Scale factor for KDE height normalization
            kde_alpha=0.99,  # Transparency of KDE fills
            kde_edge_linewidth=0.5,  # Line width for KDE edges
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
            # KDE Configuration
            "kde_resolution": "Number of points for KDE evaluation (higher = smoother curves)",
            "kde_height_scale": "Scale factor for KDE height normalization",
            "kde_alpha": "Transparency of KDE fills (0=transparent, 1=opaque)",
            "kde_edge_linewidth": "Line width for KDE curve edges",
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

    def _plot_reference_events(self, reference_events, variable):
        """Draw vertical lines and labels for reference events."""
        # Vertical lines
        self.ax.vlines(
            reference_events[variable],
            ymin=0,
            ymax=len(self._highlighted_periods),
            color=self.colors.black,
            linestyle="--",
            linewidth=self._config["reference_event_linewidth"],
            zorder=3,
        )

        # Text labels with boxes
        for reference_event in reference_events.iterrows():
            self._add_event_label(reference_event, variable)

    def _calculate_percentile_scores(self, data, reference_events, variable):
        """Calculate percentile scores for reference events across periods."""

        def _compute_variable_percentiles(s, variable):
            return pd.Series(
                {
                    period: stats.percentileofscore(
                        data.query(f"`{period}`")[variable],
                        s[variable],
                        kind="rank",
                    )
                    for period in self._highlighted_periods.keys()
                }
            )

        scores = (
            reference_events.set_index("label")
            .apply(lambda x: _compute_variable_percentiles(x, variable), axis=1)
            .T.assign(_max=101)
            .T.sort_values(by=list(self._highlighted_periods.keys())[0])
        )

        scores.loc["_max"] = 100
        scores -= scores.shift(1).fillna(0)
        return scores

    def _plot_single_kde(
        self, events, variable, period_index, max_value=None, min_value=None
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

        plot_kwargs = dict(
            color=self.colors.get_color(period_index),
            zorder=1
            + (len(self._highlighted_periods) - 1 - period_index)
            / len(self._highlighted_periods),
            alpha=self._config["kde_alpha"],
            linewidth=self._config["kde_edge_linewidth"],
            edgecolor=self.colors.black,
        )

        self.ax.fill_between(
            x_values,
            [period_index] * len(kde_values),
            kde_values,
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
        for k, score in enumerate(scores[highlighted_period].values):
            ax_text(
                x=scores_positions[k],
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

    def _plot_kde_distributions(
        self,
        data,
        variable,
        scores_reference_events,
        score_positions_reference_events,
        max_value=None,
        min_value=None,
    ):
        """Plot KDE distributions for each highlighted period."""
        for i, highlighted_period in enumerate(self._highlighted_periods.keys()):
            # Get events for this period
            _events = data.query(f"`{highlighted_period}`")

            # Plot KDE
            self._plot_single_kde(
                _events, variable, i, max_value=max_value, min_value=min_value
            )

            # Add percentage scores
            if (
                scores_reference_events is not None
                and not scores_reference_events.empty
            ):
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
    ):
        """Plot the KDE distribution of a variable over specified periods.

        Args:
            data (pd.DataFrame): DataFrame containing the data to plot.
            variable (str): The variable/column name to plot.
            periods_columns_labels (Dict[str, str]): Dictionary mapping period column names (boolean columns saying whether an event belongs to a period) to their labels.
            reference_events (Optional[pd.DataFrame]): Optional DataFrame for data of reference events (same structure as data).
            bounds (Optional[List[float]]): Optional bounds for the x-axis.
            cut_kdes (bool): Whether to cut the KDEs at the min/max values of the data.
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
        if reference_events is not None and not reference_events.empty:
            # Plot reference events with lines and labels
            self._plot_reference_events(reference_events, variable)

            # Compute scores and positions for reference data
            _scores = self._calculate_percentile_scores(
                data, reference_events, variable
            )
            _scores_positions = self._calculate_score_positions(
                reference_events, variable, min_value, max_value, _scores
            )
        else:
            _scores = None
            _scores_positions = None

        # 5. Plot KDE distributions and scores
        self._plot_kde_distributions(
            data,
            variable,
            _scores,
            _scores_positions,
            max_value=None if cut_kdes else max_value,
            min_value=None if cut_kdes else min_value,
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
