from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QComboBox, QPushButton, QMessageBox, QSlider
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import pandas as pd
import numpy as np
import seaborn as sns
from statsmodels.tsa.statespace.sarimax import SARIMAX
from datetime import datetime, timedelta


class StateAnalysis(QWidget):
    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.setup_ui()

    def setup_ui(self):
        """Set up the user interface for state-wide analysis."""
        # Load data
        self.file_path = '/Users/phoebegwimo/Library/Mobile Documents/com~apple~CloudDocs/TU Braunschweig /Semester 3/Python Lab/COVID_Prediction/time_series_covid19_confirmed_US.csv'
        self.shapefile_path = '/Users/phoebegwimo/Library/Mobile Documents/com~apple~CloudDocs/TU Braunschweig /Semester 3/Python Lab/COVID_Prediction/shapefile_data/cb_2018_us_state_500k.shp'
        self.data = pd.read_csv(self.file_path)
        self.usa_states = gpd.read_file(self.shapefile_path)
        self.unique_states = sorted(self.data['Province_State'].unique())
        self.date_columns = self.data.columns[11:]

        # Create layout
        self.layout = QVBoxLayout()

        # Title
        title_label = QLabel("State-wide Analysis")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("font-size: 18px; font-weight: bold;")
        self.layout.addWidget(title_label)

        # State Dropdown
        self.state_label = QLabel("Select a State:")
        self.layout.addWidget(self.state_label)
        self.state_dropdown = QComboBox()
        self.state_dropdown.addItems(self.unique_states)
        self.layout.addWidget(self.state_dropdown)

        # Graphical View Button
        graphical_button = QPushButton("Graphical View")
        graphical_button.setStyleSheet("background-color: blue; color: white; font-size: 14px;")
        graphical_button.clicked.connect(self.graphical_view)
        self.layout.addWidget(graphical_button)

        # Geographical View Button
        geographical_button = QPushButton("Geographical View")
        geographical_button.setStyleSheet("background-color: green; color: white; font-size: 14px;")
        geographical_button.clicked.connect(self.geographical_view)
        self.layout.addWidget(geographical_button)

        # Back Button
        back_button = QPushButton("Back")
        back_button.setStyleSheet("background-color: red; color: white; font-size: 14px;")
        back_button.clicked.connect(self.main_window.analyze_data_page)
        self.layout.addWidget(back_button)

        # Set the layout
        self.setLayout(self.layout)

    def graphical_view(self):
        """Perform SARIMA analysis and display graphical results."""
        try:
            state_name = self.state_dropdown.currentText()
            if not state_name:
                QMessageBox.warning(self, "Error", "Please select a state.")
                return

            state_data = self.data[self.data['Province_State'] == state_name]
            if state_data.empty:
                QMessageBox.warning(self, "Error", f"No data found for {state_name}.")
                return

            # Prepare data for analysis
            cumulative_cases = state_data[self.date_columns].sum().values.astype(float)
            cumulative_cases = np.nan_to_num(cumulative_cases)
            dates = pd.to_datetime(self.date_columns, format='%m/%d/%y')
            daily_new_cases = np.diff(cumulative_cases, prepend=0)

            # Configure SARIMA
            p, d, q = 1, 1, 1
            P, D, Q, s = 1, 1, 1, 7
            sarima_model = SARIMAX(daily_new_cases, order=(p, d, q), seasonal_order=(P, D, Q, s))
            sarima_fit = sarima_model.fit(disp=False)

            # Forecast future cases
            today = pd.Timestamp(datetime.today().strftime('%Y-%m-%d'))
            last_date = dates[-1]
            days_to_predict = (today - last_date).days

            future_predictions = sarima_fit.forecast(steps=days_to_predict)
            future_predictions = np.clip(future_predictions, 0, None)
            predicted_cumulative_cases = np.cumsum(future_predictions) + cumulative_cases[-1]
            future_dates = [last_date + timedelta(days=i + 1) for i in range(days_to_predict)]

            # Create the plots
            fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=100)

            # Plot cumulative total cases
            axes[0].plot(dates, cumulative_cases, label='Actual Total Cases', color='blue', alpha=0.6)
            axes[0].plot(future_dates, predicted_cumulative_cases, label='Predicted Total Cases', color='red', linestyle='--')
            axes[0].set_title(f'{state_name} - Cumulative Total Cases', fontsize=14)
            axes[0].set_xlabel('Date', fontsize=12)
            axes[0].set_ylabel('Total Cases', fontsize=12)
            axes[0].legend()
            axes[0].grid(True, linestyle='--', alpha=0.6)

            # Plot daily new cases
            axes[1].bar(dates, daily_new_cases, color='gray', alpha=0.5, label='Actual Daily New Cases', width=1.5)
            axes[1].bar(future_dates, future_predictions, color='red', alpha=0.5, label='Predicted Daily New Cases', width=1.5)
            axes[1].set_title(f'{state_name} - Daily New Cases', fontsize=14)
            axes[1].set_xlabel('Date', fontsize=12)
            axes[1].set_ylabel('New Cases', fontsize=12)
            axes[1].legend()
            axes[1].grid(True, linestyle='--', alpha=0.6)

            # Display the plots in a canvas
            canvas = FigureCanvas(fig)
            self.layout.addWidget(canvas)
            canvas.draw()

        except Exception as e:
            QMessageBox.critical(self, "Error", f"An error occurred: {str(e)}")

###################################
    def navigate_back(self):
        self.parent.analyze_data_page()  # Navigate to Analyze Data Page
        self.close()

    def geographical_view(self):
        print("Geographical View button clicked.")

        try:
            # File paths
            shapefile_path = '/Users/phoebegwimo/Library/Mobile Documents/com~apple~CloudDocs/TU Braunschweig /Semester 3/Python Lab/COVID_Prediction/shapefile_data/cb_2018_us_state_500k.shp'
            csv_file = '/Users/phoebegwimo/Library/Mobile Documents/com~apple~CloudDocs/TU Braunschweig /Semester 3/Python Lab/COVID_Prediction/time_series_covid19_confirmed_US.csv'

            # Load shapefile and COVID-19 data
            usa_states = gpd.read_file(shapefile_path)
            covid_data = pd.read_csv(csv_file)

            # Extract state names and date columns
            state_names = sorted(covid_data["Province_State"].unique())
            date_columns = covid_data.columns[11:]
            covid_data[date_columns] = covid_data[date_columns].apply(pd.to_numeric, errors="coerce").fillna(0)

            # Add dropdown for state selection
            state_label = QLabel("Select a State:", self)
            state_label.setAlignment(Qt.AlignCenter)
            state_label.setStyleSheet("font-size: 16px; margin: 10px;")
            self.layout.addWidget(state_label)

            state_dropdown = QComboBox(self)
            state_dropdown.addItems(state_names)
            self.layout.addWidget(state_dropdown)

            # Add slider for date selection
            slider_label = QLabel("Select a Date:", self)
            slider_label.setAlignment(Qt.AlignCenter)
            slider_label.setStyleSheet("font-size: 16px; margin: 10px;")
            self.layout.addWidget(slider_label)

            slider = QSlider(Qt.Horizontal, self)
            slider.setMinimum(0)
            slider.setMaximum(len(date_columns) - 1)
            slider.setValue(len(date_columns) - 1)
            slider.setTickPosition(QSlider.TicksBelow)
            slider.setTickInterval(10)
            self.layout.addWidget(slider)

            # Add canvas for displaying plots
            self.canvas = FigureCanvas(plt.figure(figsize=(14, 8)))  # Increased the canvas size
            self.layout.addWidget(self.canvas)

            def plot_geographical_view(state_name, date_index):
                selected_date = date_columns[date_index]
                fig = self.canvas.figure
                fig.clear()

                # Create subplots
                ax1 = fig.add_subplot(121)  # Heatmap on the left
                ax2 = fig.add_subplot(122)  # County progression on the right

                # Filter the state shape and data
                state_shape = usa_states[usa_states["NAME"] == state_name]
                state_covid_data = covid_data[covid_data["Province_State"] == state_name]

                if state_shape.empty or state_covid_data.empty:
                    QMessageBox.critical(self, "Error", f"No data available for {state_name}.")
                    return

                # Plot Heatmap (Left Plot)
                state_gdf = gpd.GeoDataFrame(
                    state_covid_data,
                    geometry=gpd.points_from_xy(state_covid_data.Long_, state_covid_data.Lat),
                    crs="EPSG:4326"
                )
                max_cases_last_day = state_covid_data[date_columns[-1]].max()
                state_gdf["cases"] = state_gdf[selected_date]
                heatmap_data = state_gdf[["geometry", "cases"]].dropna()
                normalized_weights = heatmap_data["cases"] / max_cases_last_day

                state_shape.boundary.plot(ax=ax1, linewidth=2, color="black")  # Thicker boundary
                sns.kdeplot(
                    x=heatmap_data.geometry.x,
                    y=heatmap_data.geometry.y,
                    weights=normalized_weights,
                    ax=ax1,
                    cmap="rainbow",
                    fill=True,
                    alpha=0.6,
                    bw_adjust=0.5
                )
                ax1.set_xlim(state_shape.total_bounds[[0, 2]])  # Zoom to the state bounds
                ax1.set_ylim(state_shape.total_bounds[[1, 3]])  # Adjust aspect ratio
                ax1.set_title(f"COVID-19 Heatmap - {state_name} on {selected_date}", fontsize=14)
                ax1.axis("off")

                # Plot Total Cases as Dots (Right Plot)
                state_shape.boundary.plot(ax=ax2, color="black", linewidth=2)  # Thicker boundary
                state_gdf.plot(
                    ax=ax2,
                    color="blue",
                    markersize=state_gdf["cases"] / 100,  # Scale marker size
                    alpha=0.7
                )
                ax2.set_xlim(state_shape.total_bounds[[0, 2]])  # Zoom to the state bounds
                ax2.set_ylim(state_shape.total_bounds[[1, 3]])  # Adjust aspect ratio
                ax2.set_title(f"Total Cases by County - {state_name} on {selected_date}", fontsize=14)
                ax2.axis("off")

                self.canvas.draw()

            # Connect dropdown and slider to update plots dynamically
            def update_plots():
                state_name = state_dropdown.currentText()
                date_index = slider.value()
                plot_geographical_view(state_name, date_index)

            state_dropdown.currentTextChanged.connect(update_plots)
            slider.valueChanged.connect(update_plots)

            # Plot initial view
            plot_geographical_view(state_dropdown.currentText(), slider.value())

        except FileNotFoundError:
            QMessageBox.critical(self, "Error", "Shapefile or dataset not found. Please ensure the files are in the correct location.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"An error occurred: {str(e)}")
