from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QComboBox, QPushButton, QMessageBox, QDateEdit, QDialog, QSlider
)
from PyQt5.QtCore import QDate, Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import geopandas as gpd
import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
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
        return self.date_picker.date().toString("yyyy-MM-dd")


class StateAnalysis(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent
        self.setWindowTitle("State-wide Analysis")
        self.setGeometry(100, 100, 1000, 700)
        self.setup_ui()
        self.selected_date = None  # To store the user-selected prediction date
        self.future_predictions = None  # To store future predictions
        self.predicted_cumulative_cases = None  # To store predicted cumulative cases

    def setup_ui(self):
        self.layout = QVBoxLayout(self)

        # Title Label
        title_label = QLabel("State-wide Analysis", self)
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("font-size: 24px; font-weight: bold; margin: 20px;")
        self.layout.addWidget(title_label)

        # State Selection Dropdown
        state_label = QLabel("Select a State:", self)
        state_label.setStyleSheet("font-size: 16px;")
        self.layout.addWidget(state_label)
        self.state_dropdown = QComboBox(self)
        self.populate_states()  # Populate dropdown with state names
        self.layout.addWidget(self.state_dropdown)

        # Graphical View Button
        graphical_button = QPushButton("Graphical View", self)
        graphical_button.setStyleSheet("font-size: 16px; background-color: blue; color: white; padding: 10px;")
        graphical_button.clicked.connect(self.graphical_view)
        self.layout.addWidget(graphical_button)

        # Geographical View Button
        geographical_button = QPushButton("Geographical View", self)
        geographical_button.setStyleSheet("font-size: 16px; background-color: green; color: white; padding: 10px;")
        geographical_button.clicked.connect(self.geographical_view)
        self.layout.addWidget(geographical_button)

        # Back Button
        back_button = QPushButton("Back", self)
        back_button.setStyleSheet("font-size: 16px; background-color: red; color: white; padding: 10px;")
        back_button.clicked.connect(self.navigate_back)
        self.layout.addWidget(back_button)

    def populate_states(self):
        """Populate the dropdown with state names from the dataset."""
        dataset_path = '/Users/phoebegwimo/Library/Mobile Documents/com~apple~CloudDocs/TU Braunschweig /Semester 3/Python Lab/COVID_Prediction/time_series_covid19_confirmed_US.csv'
        data = pd.read_csv(dataset_path)
        self.state_dropdown.addItems(sorted(data['Province_State'].unique()))

    def graphical_view(self):
        try:
            selected_state = self.state_dropdown.currentText()
            if not selected_state:
                QMessageBox.warning(self, "Error", "Please select a state.")
                return

            # Load dataset
            dataset_path = '/Users/phoebegwimo/Library/Mobile Documents/com~apple~CloudDocs/TU Braunschweig /Semester 3/Python Lab/COVID_Prediction/time_series_covid19_confirmed_US.csv'
            data = pd.read_csv(dataset_path)
            state_data = data[data['Province_State'] == selected_state]

            # Prepare data
            date_columns = data.columns[11:]
            cumulative_cases = state_data[date_columns].sum().values.astype(float)
            dates = pd.to_datetime(date_columns, format='%m/%d/%y')
            daily_new_cases = np.diff(cumulative_cases, prepend=0)

            # Prompt for prediction date
            dialog = DatePickerDialog(self)
            if dialog.exec() != QDialog.Accepted:
                return  # Exit if the user cancels
            self.selected_date = pd.to_datetime(dialog.get_selected_date())

            # Validate prediction date
            last_date = dates[-1]
            if self.selected_date <= last_date:
                QMessageBox.warning(self, "Error", "Selected date must be in the future.")
                return

            # Prediction setup
            days_to_predict = (self.selected_date - last_date).days
            sarima_model = SARIMAX(daily_new_cases, order=(1, 1, 1), seasonal_order=(1, 1, 1, 7))
            sarima_fit = sarima_model.fit(disp=False)

            # Predictions
            self.future_predictions = sarima_fit.forecast(steps=days_to_predict)
            self.future_predictions = np.clip(self.future_predictions, 0, None)
            self.predicted_cumulative_cases = np.cumsum(self.future_predictions) + cumulative_cases[-1]
            future_dates = [last_date + timedelta(days=i + 1) for i in range(days_to_predict)]

            # Plot results
            fig, axes = plt.subplots(1, 2, figsize=(16, 6), dpi=100)
            axes[0].plot(dates, cumulative_cases, label='Actual Total Cases', color='blue', alpha=0.6)
            axes[0].plot(future_dates, self.predicted_cumulative_cases, label='Predicted Total Cases', color='red', linestyle='--')
            axes[0].set_title(f'{selected_state} - Cumulative Cases (SARIMA)')
            axes[0].legend()
            axes[0].grid()

            axes[1].bar(dates, daily_new_cases, label='Actual New Cases', color='gray', alpha=0.5)
            axes[1].bar(future_dates, self.future_predictions, label='Predicted New Cases', color='red', alpha=0.5)
            axes[1].set_title(f'{selected_state} - Daily New Cases (SARIMA)')
            axes[1].legend()
            axes[1].grid()

            canvas = FigureCanvas(fig)
            self.layout.addWidget(canvas)
            canvas.draw()

            # Add Make Prediction button
            predict_button = QPushButton("Make Prediction", self)
            predict_button.setStyleSheet("font-size: 16px; background-color: orange; color: white; padding: 10px;")
            predict_button.clicked.connect(self.make_prediction)
            self.layout.addWidget(predict_button)

        except Exception as e:
            QMessageBox.critical(self, "Error", f"An error occurred: {str(e)}")

    def make_prediction(self):
        if self.selected_date is None or self.future_predictions is None:
            QMessageBox.warning(self, "Error", "Generate predictions first in Graphical View.")
            return

        predicted_new_cases = self.future_predictions[-1]
        predicted_total_cases = self.predicted_cumulative_cases[-1]

        QMessageBox.information(
            self,
            "Prediction Results",
            f"Predicted new cases for {self.state_dropdown.currentText()} on {self.selected_date.strftime('%Y-%m-%d')}: {int(predicted_new_cases)}\n"
            f"Predicted total cases: {int(predicted_total_cases)}"
        )


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

    def navigate_back(self):
        """Navigate back to the main menu or parent view."""
        self.parent.analyze_data_page() if self.parent else self.close()
