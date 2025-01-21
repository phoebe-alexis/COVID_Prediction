from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QComboBox, QPushButton, QMessageBox, QSlider, QDialog, QDateEdit, QHBoxLayout
from PyQt5.QtCore import Qt, QDate
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from datetime import datetime, timedelta
import seaborn as sns



class DatePickerDialog(QDialog):
    """Dialog for selecting a prediction date."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Select a Prediction Date")
        self.setGeometry(300, 200, 400, 150)
        
        layout = QVBoxLayout(self)

        # Instruction Label
        instruction_label = QLabel("Please select a prediction date:")
        layout.addWidget(instruction_label)

        # Date Picker
        self.date_picker = QDateEdit(self)
        self.date_picker.setCalendarPopup(True)
        self.date_picker.setDate(QDate.currentDate())
        self.date_picker.setDisplayFormat("yyyy-MM-dd")
        layout.addWidget(self.date_picker)

        # OK Button
        ok_button = QPushButton("OK", self)
        ok_button.clicked.connect(self.accept)
        layout.addWidget(ok_button)

        # Cancel Button
        cancel_button = QPushButton("Cancel", self)
        cancel_button.clicked.connect(self.reject)
        layout.addWidget(cancel_button)

    def get_selected_date(self):
        """Return the selected date as a string."""
        return self.date_picker.date().toString("yyyy-MM-dd")

class CountyAnalysis(QWidget):
    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent  # Store the parent object
        self.setWindowTitle("County-wide Analysis")
        self.setGeometry(100, 100, 1000, 700)

        # Initialize attributes
        self.file_path = '/Users/phoebegwimo/Library/Mobile Documents/com~apple~CloudDocs/TU Braunschweig /Semester 3/Python Lab/COVID_Prediction/time_series_covid19_confirmed_US.csv'
        self.shapefile_path = '/Users/phoebegwimo/Library/Mobile Documents/com~apple~CloudDocs/TU Braunschweig /Semester 3/Python Lab/COVID_Prediction/shapefile_data/cb_2018_us_state_500k.shp'

        self.data = pd.read_csv(self.file_path)
        self.usa_states = gpd.read_file(self.shapefile_path)
        self.date_columns = self.data.columns[11:]
        self.unique_states = sorted(self.data['Province_State'].unique())
        self.canvas = None  # Initialize canvas as None

        # Set up the UI
        self.setup_ui()

    def setup_ui(self):
        """Set up the basic UI for county-wide analysis."""
        layout = QVBoxLayout(self)

        # Title
        title_label = QLabel("County-wide Analysis", self)
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("font-size: 24px; font-weight: bold; margin: 20px;")
        layout.addWidget(title_label)

        # State Dropdown
        self.state_label = QLabel("Select a State:", self)
        layout.addWidget(self.state_label)
        self.state_dropdown = QComboBox(self)
        self.state_dropdown.addItems(self.unique_states)
        self.state_dropdown.currentTextChanged.connect(self.update_counties)
        layout.addWidget(self.state_dropdown)

        # County Dropdown
        self.county_label = QLabel("Select a County:", self)
        layout.addWidget(self.county_label)
        self.county_dropdown = QComboBox(self)
        layout.addWidget(self.county_dropdown)

        # Buttons
        graphical_button = QPushButton("Graphical View", self)
        graphical_button.setStyleSheet("font-size: 16px; background-color: blue; color: white; padding: 10px;")
        graphical_button.clicked.connect(self.graphical_view)
        layout.addWidget(graphical_button)

        geographical_button = QPushButton("Geographical View", self)
        geographical_button.setStyleSheet("font-size: 16px; background-color: green; color: white; padding: 10px;")
        geographical_button.clicked.connect(self.geographical_view)
        layout.addWidget(geographical_button)

        back_button = QPushButton("Back", self)
        back_button.setStyleSheet("font-size: 16px; background-color: red; color: white; padding: 10px;")
        back_button.clicked.connect(self.parent.home_page)
        layout.addWidget(back_button)

        self.setLayout(layout)

    def update_counties(self, state_name):
        """Update the county dropdown based on the selected state."""
        counties = sorted(self.data[self.data['Province_State'] == state_name]['Admin2'].unique())
        self.county_dropdown.clear()
        self.county_dropdown.addItems(counties)


    def graphical_view(self):
        """Graphical View: Perform SARIMA analysis and display results for a selected county."""
        try:
            # Ensure a state and county are selected
            state_name = self.state_dropdown.currentText()
            county_name = self.county_dropdown.currentText()

            if not state_name or not county_name:
                QMessageBox.warning(self, "Error", "Please select both a state and a county.")
                return

            # Filter data for the selected county
            county_data = self.data[(self.data['Province_State'] == state_name) & (self.data['Admin2'] == county_name)]

            if county_data.empty:
                QMessageBox.warning(self, "Error", f"No data found for {county_name}, {state_name}.")
                return

            # Extract date columns and prepare data
            date_columns = self.date_columns
            cumulative_cases = county_data[date_columns].sum().values.astype(float)
            dates = pd.to_datetime(date_columns, format='%m/%d/%y')
            daily_new_cases = np.diff(cumulative_cases, prepend=0)

            # Open a DatePickerDialog for the user to select a prediction date
            date_dialog = DatePickerDialog(self)
            if date_dialog.exec_() != QDialog.Accepted:
                return  # Exit if the dialog is canceled
            selected_date = date_dialog.get_selected_date()
            self.selected_date = pd.to_datetime(selected_date)

            # Validate the selected date
            last_date = dates[-1]
            if self.selected_date <= last_date:
                QMessageBox.warning(self, "Error", "Selected date must be in the future.")
                return

            # Days to predict
            days_to_predict = (self.selected_date - last_date).days

            # Configure and fit SARIMA model
            sarima_model = SARIMAX(daily_new_cases, order=(1, 1, 1), seasonal_order=(1, 1, 1, 7))
            sarima_fit = sarima_model.fit(disp=False)

            # Generate future predictions
            self.future_predictions = sarima_fit.forecast(steps=days_to_predict)
            self.future_predictions = np.clip(self.future_predictions, 0, None)
            self.predicted_cumulative_cases = np.cumsum(self.future_predictions) + cumulative_cases[-1]
            future_dates = [last_date + timedelta(days=i + 1) for i in range(days_to_predict)]

            # Create and display plots
            fig, axes = plt.subplots(1, 2, figsize=(16, 6), dpi=100)

            # Plot cumulative total cases
            axes[0].plot(dates, cumulative_cases, label='Actual Total Cases', color='blue', alpha=0.6)
            axes[0].plot(future_dates, self.predicted_cumulative_cases, label='Predicted Total Cases', color='red', linestyle='--')
            axes[0].set_title(f'Cumulative Total Cases in {county_name}, {state_name}', fontsize=16)
            axes[0].set_xlabel('Date', fontsize=14)
            axes[0].set_ylabel('Total Cases', fontsize=14)
            axes[0].legend()
            axes[0].grid(True, linestyle='--', alpha=0.6)

            # Plot daily new cases
            axes[1].bar(dates, daily_new_cases, label='Actual Daily New Cases', color='gray', alpha=0.5, width=2)
            axes[1].bar(future_dates, self.future_predictions, label='Predicted Daily New Cases', color='red', alpha=0.5, width=2)
            axes[1].set_title(f'Daily New Cases in {county_name}, {state_name}', fontsize=16)
            axes[1].set_xlabel('Date', fontsize=14)
            axes[1].set_ylabel('New Cases', fontsize=14)
            axes[1].legend()
            axes[1].grid(True, linestyle='--', alpha=0.6)

            plt.tight_layout()

            # Display the plot
            if self.canvas:
                self.layout().removeWidget(self.canvas)
                self.canvas.deleteLater()
            self.canvas = FigureCanvas(fig)
            self.layout().addWidget(self.canvas)
            self.canvas.draw()

            # Add "Make Prediction" button
            predict_button = QPushButton("Make Prediction", self)
            predict_button.setStyleSheet("font-size: 16px; background-color: orange; color: white; padding: 10px;")
            predict_button.clicked.connect(self.make_prediction)
            self.layout().addWidget(predict_button)

        except Exception as e:
            QMessageBox.critical(self, "Error", f"An error occurred: {str(e)}")


    def make_prediction(self):
        """Display numerical results for the user-selected date."""
        if self.selected_date is None or self.future_predictions is None:
            QMessageBox.warning(self, "Error", "Please generate predictions first by selecting a date in Graphical View.")
            return

        # Get the last prediction (for the user-selected date)
        predicted_new_cases = self.future_predictions[-1]
        predicted_total_cases = self.predicted_cumulative_cases[-1]
        # Fetch the selected county name
        county_name = self.county_dropdown.currentText()


        QMessageBox.information(
            self,
            "Prediction Results",
            f"Predicted new cases for {self.county_dropdown.currentText()} on {self.selected_date.strftime('%Y-%m-%d')}: {int(predicted_new_cases)}\n"
            f"Predicted total cases for {self.selected_date.strftime('%Y-%m-%d')}: {int(predicted_total_cases)}"
        )

    def geographical_view(self):
        """Geographical View: Plot heatmap and scatter plot for selected state and county."""
        try:
            # File paths
            shapefile_path = self.shapefile_path
            csv_file = self.file_path

            # Load shapefile and COVID-19 data
            usa_states = gpd.read_file(shapefile_path)
            covid_data = pd.read_csv(csv_file)

            # Extract state names and date columns
            date_columns = covid_data.columns[11:]
            covid_data[date_columns] = covid_data[date_columns].apply(pd.to_numeric, errors="coerce").fillna(0)

            # Add slider for date selection
            slider_label = QLabel("Select a Date:", self)
            slider_label.setAlignment(Qt.AlignCenter)
            slider_label.setStyleSheet("font-size: 16px; margin: 10px;")
            self.layout().addWidget(slider_label)

            slider = QSlider(Qt.Horizontal, self)
            slider.setMinimum(0)
            slider.setMaximum(len(date_columns) - 1)
            slider.setValue(len(date_columns) - 1)
            slider.setTickPosition(QSlider.TicksBelow)
            slider.setTickInterval(10)
            self.layout().addWidget(slider)

            # Add canvas for displaying plots
            if self.canvas is None:
                self.canvas = FigureCanvas(plt.figure(figsize=(14, 8)))  # Large canvas for the plots
                self.layout().addWidget(self.canvas)

            def plot_geographical_view(date_index):
                selected_date = date_columns[date_index]
                state_name = self.state_dropdown.currentText()
                county_name = self.county_dropdown.currentText()

                # Prepare data for the selected state
                state_shape = usa_states[usa_states["NAME"] == state_name]
                state_covid_data = covid_data[covid_data["Province_State"] == state_name].copy()

                if state_shape.empty or state_covid_data.empty:
                    QMessageBox.critical(self, "Error", f"No data available for {state_name}.")
                    return

                # Add total cases for the selected date
                state_covid_data["cases"] = state_covid_data[selected_date]

                # Separate data for the selected county
                selected_county_data = state_covid_data[state_covid_data["Admin2"] == county_name]
                other_counties_data = state_covid_data[state_covid_data["Admin2"] != county_name]

                # Create GeoDataFrames for selected and other counties
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

                # Plot
                fig = self.canvas.figure
                fig.clear()

                ax1 = fig.add_subplot(121)  # Heatmap
                ax2 = fig.add_subplot(122)  # Scatter plot

                # Heatmap Plot
                heatmap_data = gpd.GeoDataFrame(
                    state_covid_data,
                    geometry=gpd.points_from_xy(state_covid_data.Long_, state_covid_data.Lat),
                    crs="EPSG:4326"
                )
                state_shape.boundary.plot(ax=ax1, linewidth=2, color="black")
                sns.kdeplot(
                    x=heatmap_data.geometry.x,
                    y=heatmap_data.geometry.y,
                    weights=heatmap_data["cases"],
                    ax=ax1,
                    cmap="rainbow",
                    fill=True,
                    alpha=0.6,
                    bw_adjust=0.5
                )
                ax1.set_xlim(state_shape.total_bounds[[0, 2]])
                ax1.set_ylim(state_shape.total_bounds[[1, 3]])
                ax1.set_title(f"COVID-19 Heatmap - {state_name} on {selected_date}", fontsize=14)
                ax1.axis("off")

                # Scatter Plot
                state_shape.boundary.plot(ax=ax2, linewidth=2, color="black")  # Plot state boundary

                # Plot other counties in blue
                gdf_other.plot(
                    ax=ax2,
                    color="blue",
                    markersize=gdf_other["cases"] / 100,
                    alpha=0.7
                )

                # Plot the selected county in red
                gdf_selected.plot(
                    ax=ax2,
                    color="red",
                    markersize=gdf_selected["cases"] / 100,
                    alpha=0.9
                )

                # Customize the scatter plot
                ax2.set_xlim(state_shape.total_bounds[[0, 2]])
                ax2.set_ylim(state_shape.total_bounds[[1, 3]])
                ax2.set_title(f"Total Cases by County - {state_name} on {selected_date}", fontsize=14)
                ax2.axis("off")

                # Update the canvas
                self.canvas.draw()

            # Connect slider to dynamically update the plot
            slider.valueChanged.connect(lambda value: plot_geographical_view(value))

            # Plot the initial view
            plot_geographical_view(slider.value())

        except FileNotFoundError:
            QMessageBox.critical(self, "Error", "Shapefile or dataset not found. Please ensure the files are in the correct location.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"An error occurred: {str(e)}")
