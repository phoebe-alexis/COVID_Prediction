from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QPushButton, QMessageBox, QDateEdit, QDialog, QSlider
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


class CountryAnalysis(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent
        self.setWindowTitle("Country-wide Analysis")
        self.setGeometry(100, 100, 1000, 700)
        self.setup_ui()
        self.selected_date = None  # To store the user-selected prediction date
        self.future_predictions = None  # To store future predictions
        self.predicted_cumulative_cases = None  # To store predicted cumulative cases

    def setup_ui(self):
        self.layout = QVBoxLayout(self)

        # Title Label
        title_label = QLabel("Country-wide Analysis", self)
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("font-size: 24px; font-weight: bold; margin: 20px;")
        self.layout.addWidget(title_label)

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

    def graphical_view(self):
        try:
            # File path for the dataset
            file_path = '/Users/phoebegwimo/Library/Mobile Documents/com~apple~CloudDocs/TU Braunschweig /Semester 3/Python Lab/COVID_Prediction/time_series_covid19_confirmed_US.csv'
            data = pd.read_csv(file_path)

            # Extract date columns and calculate cumulative cases and daily new cases
            date_columns = data.columns[11:]
            cumulative_cases = data[date_columns].sum().values.astype(float)
            dates = pd.to_datetime(date_columns, format='%m/%d/%y')
            daily_new_cases = np.diff(cumulative_cases, prepend=0)

            # Prompt the user to select a date for predictions
            dialog = DatePickerDialog(self)
            if dialog.exec() != QDialog.Accepted:
                return  # Exit if the user cancels the date selection
            self.selected_date = pd.to_datetime(dialog.get_selected_date())

            # Validate selected date
            last_date = dates[-1]
            if self.selected_date <= last_date:
                QMessageBox.warning(self, "Error", "Selected date must be in the future.")
                return

            # Calculate days to predict
            days_to_predict = (self.selected_date - last_date).days

            # Configure SARIMA model
            sarima_model = SARIMAX(daily_new_cases, order=(1, 1, 1), seasonal_order=(1, 1, 1, 7))
            sarima_fit = sarima_model.fit(disp=False)

            # Predict future cases
            self.future_predictions = sarima_fit.forecast(steps=days_to_predict)
            self.future_predictions = np.clip(self.future_predictions, 0, None)
            self.predicted_cumulative_cases = np.cumsum(self.future_predictions) + cumulative_cases[-1]
            future_dates = [last_date + timedelta(days=i + 1) for i in range(days_to_predict)]

            # Create plots
            fig, axes = plt.subplots(1, 2, figsize=(16, 6), dpi=100)

            # Cumulative total cases plot
            axes[0].plot(dates, cumulative_cases, label='Actual Total Cases', color='blue', alpha=0.6)
            axes[0].plot(future_dates, self.predicted_cumulative_cases, label='Predicted Total Cases', color='red', linestyle='--')
            axes[0].set_title('Cumulative Total Cases (SARIMA)', fontsize=16)
            axes[0].set_xlabel('Date', fontsize=14)
            axes[0].set_ylabel('Total Cases', fontsize=14)
            axes[0].legend()
            axes[0].grid(True, linestyle='--', alpha=0.6)

            # Daily new cases plot
            axes[1].bar(dates, daily_new_cases, label='Actual Daily New Cases', color='gray', alpha=0.5, width=2)
            axes[1].bar(future_dates, self.future_predictions, label='Predicted Daily New Cases', color='red', alpha=0.5, width=2)
            axes[1].set_title('Daily New Cases (SARIMA)', fontsize=16)
            axes[1].set_xlabel('Date', fontsize=14)
            axes[1].set_ylabel('New Cases', fontsize=14)
            axes[1].legend()
            axes[1].grid(True, linestyle='--', alpha=0.6)

            plt.tight_layout()

            # Display the plot
            canvas = FigureCanvas(fig)
            self.layout.addWidget(canvas)
            canvas.draw()

            # Add the Make Prediction button
            predict_button = QPushButton("Make Prediction", self)
            predict_button.setStyleSheet("font-size: 16px; background-color: orange; color: white; padding: 10px;")
            predict_button.clicked.connect(self.make_prediction)
            self.layout.addWidget(predict_button)

        except FileNotFoundError:
            QMessageBox.critical(self, "Error", "Dataset not found. Please ensure the file is in the correct location.")
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

        QMessageBox.information(
            self,
            "Prediction Results",
            f"Predicted new cases for the US on {self.selected_date.strftime('%Y-%m-%d')}: {int(predicted_new_cases)}\n"
            f"Predicted total cases for the US on {self.selected_date.strftime('%Y-%m-%d')}: {int(predicted_total_cases)}"
        )

    

    def geographical_view(self):
        print("Geographical View button clicked.")  # Debugging

        try:
            # File paths
            shapefile_path = '/Users/phoebegwimo/Library/Mobile Documents/com~apple~CloudDocs/TU Braunschweig /Semester 3/Python Lab/COVID_Prediction/shapefile_data/cb_2018_us_state_500k.shp'
            csv_file = '/Users/phoebegwimo/Library/Mobile Documents/com~apple~CloudDocs/TU Braunschweig /Semester 3/Python Lab/COVID_Prediction/time_series_covid19_confirmed_US.csv'

            # Load shapefile and COVID-19 data
            usa_states = gpd.read_file(shapefile_path)
            covid_data = pd.read_csv(csv_file)

            # Filter Mainland USA states
            mainland_states = [
                "AL", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA",
                "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD",
                "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH",
                "NJ", "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA",
                "RI", "SC", "SD", "TN", "TX", "UT", "VT", "VA", "WA",
                "WV", "WI", "WY"
            ]
            filtered_usa = usa_states[usa_states["STUSPS"].isin(mainland_states)]

            # Create GeoDataFrame for COVID-19 data with coordinates
            gdf_cities = gpd.GeoDataFrame(
                covid_data,
                geometry=gpd.points_from_xy(covid_data.Long_, covid_data.Lat),
                crs="EPSG:4326"
            )
            filtered_covid_data = gpd.sjoin(gdf_cities, filtered_usa, how="inner", predicate="within")

            # Extract date columns
            date_columns = covid_data.columns[11:]  # Assuming date columns start from column 12
            covid_data[date_columns] = covid_data[date_columns].apply(pd.to_numeric, errors="coerce").fillna(0)
            max_cases_last_day = covid_data[date_columns[-1]].max()

            # Add slider for date selection
            slider_label = QLabel("Select a Date:", self)
            slider_label.setAlignment(Qt.AlignCenter)
            slider_label.setStyleSheet("font-size: 16px; margin: 10px;")
            self.layout.addWidget(slider_label)

            slider = QSlider(Qt.Horizontal, self)
            slider.setMinimum(0)
            slider.setMaximum(len(date_columns) - 1)
            slider.setValue(len(date_columns) - 1)  # Default to the last date
            slider.setTickPosition(QSlider.TicksBelow)
            slider.setTickInterval(10)  # Adjust interval for ticks
            self.layout.addWidget(slider)

            # Add canvas for displaying heatmap
            self.canvas = FigureCanvas(plt.figure(figsize=(8, 6)))
            self.layout.addWidget(self.canvas)

            # Function to plot heatmap
            def plot_heatmap(date_index):
                selected_date = date_columns[date_index]
                fig = self.canvas.figure
                fig.clear()  # Clear the existing figure
                ax = fig.add_subplot(111)  # Create a single subplot

                # Plot state boundaries
                filtered_usa.boundary.plot(ax=ax, linewidth=1, color="black")

                # Prepare COVID-19 data for the selected date
                filtered_covid_data["cases"] = filtered_covid_data[selected_date]
                heatmap_data = filtered_covid_data[["geometry", "cases"]].dropna()
                normalized_weights = heatmap_data["cases"] / max_cases_last_day

                sns.kdeplot(
                    x=heatmap_data.geometry.x,
                    y=heatmap_data.geometry.y,
                    weights=normalized_weights,
                    ax=ax,
                    cmap="rainbow",
                    fill=True,
                    alpha=0.6,
                    bw_adjust=0.5
                )

                # Add colorbar and title
                norm = Normalize(vmin=0, vmax=max_cases_last_day)
                sm = ScalarMappable(cmap="rainbow", norm=norm)
                sm.set_array([])
                cbar = fig.colorbar(sm, ax=ax, orientation="vertical", fraction=0.03, pad=0.02)
                cbar.set_label("Number of Cases", fontsize=12)

                ax.set_title(f"COVID-19 Case Heatmap - {selected_date}", fontsize=16)
                ax.axis("off")
                self.canvas.draw()


            # Connect the slider to update the heatmap dynamically
            slider.valueChanged.connect(lambda value: plot_heatmap(value))

            # Plot initial heatmap
            plot_heatmap(slider.value())

        except FileNotFoundError:
            QMessageBox.critical(self, "Error", "Shapefile or dataset not found. Please ensure the files are in the correct location.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"An error occurred: {e}")

             

    def navigate_back(self):
        self.parent.analyze_data_page() if self.parent else self.close()


class DatePickerDialog(QDialog):
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
