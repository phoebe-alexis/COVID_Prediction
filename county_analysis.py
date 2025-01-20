from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QComboBox, QPushButton, QMessageBox, QSlider
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from datetime import datetime, timedelta


class CountyAnalysis(QWidget):
    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.setup_ui()

    def setup_ui(self):
        """Set up the user interface for county analysis."""
        # Load data
        self.file_path = '/Users/phoebegwimo/Library/Mobile Documents/com~apple~CloudDocs/TU Braunschweig /Semester 3/Python Lab/COVID_Prediction/time_series_covid19_confirmed_US.csv'
        self.shapefile_path = '/Users/phoebegwimo/Library/Mobile Documents/com~apple~CloudDocs/TU Braunschweig /Semester 3/Python Lab/COVID_Prediction/shapefile_data/cb_2018_us_state_500k.shp'
        self.data = pd.read_csv(self.file_path)
        self.usa_states = gpd.read_file(self.shapefile_path)
        self.date_columns = self.data.columns[11:]
        self.unique_states = sorted(self.data['Province_State'].unique())

        # Create layout
        self.layout = QVBoxLayout()

        # Title
        title_label = QLabel("County-wide Analysis")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("font-size: 18px; font-weight: bold;")
        self.layout.addWidget(title_label)

        # State Dropdown
        self.state_label = QLabel("Select a State:")
        self.layout.addWidget(self.state_label)
        self.state_dropdown = QComboBox()
        self.state_dropdown.addItems(self.unique_states)
        self.state_dropdown.currentTextChanged.connect(self.update_counties)
        self.layout.addWidget(self.state_dropdown)

        # County Dropdown
        self.county_label = QLabel("Select a County:")
        self.layout.addWidget(self.county_label)
        self.county_dropdown = QComboBox()
        self.layout.addWidget(self.county_dropdown)

        # Slider for Date Selection
        slider_label = QLabel("Select a Date:")
        slider_label.setAlignment(Qt.AlignCenter)
        slider_label.setStyleSheet("font-size: 14px;")
        self.layout.addWidget(slider_label)

        self.slider = QSlider(Qt.Horizontal, self)
        self.slider.setMinimum(0)
        self.slider.setMaximum(len(self.date_columns) - 1)
        self.slider.setValue(len(self.date_columns) - 1)  # Default to the last date
        self.slider.setTickPosition(QSlider.TicksBelow)
        self.slider.setTickInterval(10)
        self.slider.valueChanged.connect(self.update_plots)
        self.layout.addWidget(self.slider)

        # Canvas for displaying plots
        self.canvas = FigureCanvas(plt.figure(figsize=(10, 6)))
        self.layout.addWidget(self.canvas)

        # Buttons
        graphical_button = QPushButton("Graphical View")
        graphical_button.setStyleSheet("background-color: blue; color: white; font-size: 14px;")
        graphical_button.clicked.connect(self.graphical_view)
        self.layout.addWidget(graphical_button)

        geographical_button = QPushButton("Geographical View")
        geographical_button.setStyleSheet("background-color: green; color: white; font-size: 14px;")
        geographical_button.clicked.connect(self.geographical_view)
        self.layout.addWidget(geographical_button)

        back_button = QPushButton("Back")
        back_button.setStyleSheet("background-color: red; color: white; font-size: 14px;")
        back_button.clicked.connect(self.main_window.analyze_data_page)
        self.layout.addWidget(back_button)

        # Set the layout
        self.setLayout(self.layout)

        # Initialize county dropdown
        self.update_counties(self.state_dropdown.currentText())

    def update_counties(self, state_name):
        """Update the county dropdown based on the selected state."""
        counties = sorted(self.data[self.data['Province_State'] == state_name]['Admin2'].unique())
        self.county_dropdown.clear()
        self.county_dropdown.addItems(counties)
        self.update_plots()

    def update_plots(self):
        """Update the currently displayed plots based on the selected state, county, and date."""
        state_name = self.state_dropdown.currentText()
        county_name = self.county_dropdown.currentText()
        date_index = self.slider.value()

        if hasattr(self, 'current_view'):
            if self.current_view == 'graphical':
                self.graphical_view(state_name, county_name)
            elif self.current_view == 'geographical':
                self.geographical_view(state_name, county_name, date_index)


#####################################
    def graphical_view(self, state_name=None, county_name=None):
        """Graphical View: Perform SARIMA analysis and display results."""
        self.current_view = 'graphical'

        try:
            state_name = state_name or self.state_dropdown.currentText()
            county_name = county_name or self.county_dropdown.currentText()

            if not state_name or not county_name:
                QMessageBox.warning(self, "Error", "Please select a state and county.")
                return

            # Filter data
            county_data = self.data[(self.data['Province_State'] == state_name) & (self.data['Admin2'] == county_name)]

            if county_data.empty:
                QMessageBox.warning(self, "Error", f"No data found for {county_name}, {state_name}.")
                return

            # Prepare data for analysis
            cumulative_cases = county_data[self.date_columns].sum().values.astype(float)
            cumulative_cases = np.nan_to_num(cumulative_cases)
            dates = pd.to_datetime(self.date_columns, format='%m/%d/%y')
            daily_new_cases = np.diff(cumulative_cases, prepend=0)

            # Configure SARIMA parameters
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
            self.canvas.figure.set_size_inches(12, 6)  # Set the figure size
            self.canvas.figure.clear()  # Clear the figure
            axes = self.canvas.figure.subplots(1, 2)  # Create subplots

            # Plot cumulative total cases
            axes[0].plot(dates, cumulative_cases, label='Actual Total Cases', color='blue', alpha=0.6)
            axes[0].plot(future_dates, predicted_cumulative_cases, label='Predicted Total Cases', color='red', linestyle='--')
            axes[0].set_title(f'{county_name}, {state_name} - Cumulative Total Cases', fontsize=14)
            axes[0].set_xlabel('Date', fontsize=12)
            axes[0].set_ylabel('Total Cases', fontsize=12)
            axes[0].legend()
            axes[0].grid(True, linestyle='--', alpha=0.6)

            # Plot daily new cases
            axes[1].bar(dates, daily_new_cases, color='gray', alpha=0.5, label='Actual Daily New Cases', width=1.5)
            axes[1].bar(future_dates, future_predictions, color='red', alpha=0.5, label='Predicted Daily New Cases', width=1.5)
            axes[1].set_title(f'{county_name}, {state_name} - Daily New Cases', fontsize=14)
            axes[1].set_xlabel('Date', fontsize=12)
            axes[1].set_ylabel('New Cases', fontsize=12)
            axes[1].legend()
            axes[1].grid(True, linestyle='--', alpha=0.6)

            self.canvas.draw()

        except Exception as e:
            QMessageBox.critical(self, "Error", f"An error occurred: {str(e)}")

    def navigate_back(self):
        self.parent.analyze_data_page()  # Navigate to Analyze Data Page
        self.close()
        

    def geographical_view(self, state_name=None, county_name=None, date_index=None):
        """Geographical View: Show total cases by county with the selected county highlighted."""
        self.current_view = 'geographical'

        try:
            state_name = state_name or self.state_dropdown.currentText()
            county_name = county_name or self.county_dropdown.currentText()
            date_index = date_index or self.slider.value()
            date = self.date_columns[date_index]

            # Filter data for the selected state and date
            state_data = self.data[self.data['Province_State'] == state_name]
            state_shape = self.usa_states[self.usa_states['NAME'] == state_name]

            if state_shape.empty or state_data.empty:
                QMessageBox.warning(self, "Error", f"No data available for {state_name}.")
                return

            # Use total cases for the selected date
            state_data['total_cases'] = state_data[date]

            # Separate the selected county
            selected_county_data = state_data[state_data['Admin2'] == county_name]
            other_counties_data = state_data[state_data['Admin2'] != county_name]

            # Create GeoDataFrames
            gdf_selected = gpd.GeoDataFrame(
                selected_county_data,
                geometry=gpd.points_from_xy(selected_county_data.Long_, selected_county_data.Lat),
                crs="EPSG:4326"
            )
            gdf_other = gpd.GeoDataFrame(
                other_counties_data,
                geometry=gpd.points_from_xy(other_counties_data.Long_, other_counties_data.Lat),
                crs="EPSG:4326"
            )

            # Clear the figure and set consistent size
            self.canvas.figure.clear()
            ax = self.canvas.figure.add_subplot(111)

            # Plot the state boundary
            state_shape.boundary.plot(ax=ax, color="black", linewidth=1)

            # Plot other counties in blue
            gdf_other.plot(
                ax=ax,
                color="blue",
                markersize=gdf_other['total_cases'] / 100,
                alpha=0.7
            )

            # Plot the selected county in red
            gdf_selected.plot(
                ax=ax,
                color="red",
                markersize=gdf_selected['total_cases'] / 100,
                alpha=0.7
            )

            # Add title
            ax.set_title(f"Total COVID-19 Cases by County - {state_name} on {date}", fontsize=16)

            # Adjust map aspect and extent
            ax.axis("off")
            ax.set_aspect('auto')
            bounds = state_shape.total_bounds
            ax.set_xlim(bounds[0], bounds[2])
            ax.set_ylim(bounds[1], bounds[3])

            # Draw the canvas
            self.canvas.draw()

        except Exception as e:
            QMessageBox.critical(self, "Error", f"An error occurred: {str(e)}")
