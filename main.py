from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget, QLabel
from country_analysis import CountryAnalysis
from state_analysis import StateAnalysis
from county_analysis import CountyAnalysis


class MainApp(QMainWindow):
    def __init__(self):
        super().__init__()

        # Set main window properties
        self.setWindowTitle("COVID-19 Spread Prediction")
        self.setGeometry(100, 100, 800, 600)

        # Initialize the home page
        self.home_page()

    def home_page(self):
        """Display the Home Page with navigation options."""
        central_widget = QWidget()
        layout = QVBoxLayout()

        title_label = QLabel("COVID-19 Spread Prediction App")
        title_label.setStyleSheet("font-size: 20px; font-weight: bold; text-align: center;")
        layout.addWidget(title_label)

        analyze_data_button = QPushButton("Analyze Data")
        analyze_data_button.clicked.connect(self.analyze_data_page)
        layout.addWidget(analyze_data_button)

        make_prediction_button = QPushButton("Make Prediction")
        make_prediction_button.clicked.connect(self.show_placeholder_message)
        layout.addWidget(make_prediction_button)

        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

    def analyze_data_page(self):
        """Display the Analyze Data Page with analysis options."""
        central_widget = QWidget()
        layout = QVBoxLayout()

        title_label = QLabel("Analyze Data")
        title_label.setStyleSheet("font-size: 20px; font-weight: bold; text-align: center;")
        layout.addWidget(title_label)

        # Navigation Buttons
        country_analysis_button = QPushButton("Country-wide Analysis")
        country_analysis_button.clicked.connect(self.country_analysis_page)
        layout.addWidget(country_analysis_button)

        state_analysis_button = QPushButton("State-wide Analysis")
        state_analysis_button.clicked.connect(self.state_analysis_page)
        layout.addWidget(state_analysis_button)

        county_analysis_button = QPushButton("County-wide Analysis")
        county_analysis_button.clicked.connect(self.county_analysis_page)
        layout.addWidget(county_analysis_button)

        # Back button for Home Page
        back_button = QPushButton("Back")
        back_button.clicked.connect(self.home_page)
        layout.addWidget(back_button)

        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

    def country_analysis_page(self):
        """Navigate to the Country Analysis page."""
        self.country_analysis = CountryAnalysis(self)
        self.setCentralWidget(self.country_analysis)

    def state_analysis_page(self):
        """Navigate to the State Analysis page."""
        self.state_analysis = StateAnalysis(self)
        self.setCentralWidget(self.state_analysis)

    def county_analysis_page(self):
        """Navigate to the County Analysis page."""
        self.county_analysis = CountyAnalysis(self)
        self.setCentralWidget(self.county_analysis)

    def show_placeholder_message(self):
        """Show placeholder message for 'Make Prediction'."""
        central_widget = QWidget()
        layout = QVBoxLayout()

        placeholder_label = QLabel("Make Prediction functionality is not implemented yet.")
        placeholder_label.setStyleSheet("font-size: 16px; color: red; text-align: center;")
        layout.addWidget(placeholder_label)

        back_button = QPushButton("Back")
        back_button.clicked.connect(self.home_page)
        layout.addWidget(back_button)

        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)


if __name__ == "__main__":
    app = QApplication([])
    main_window = MainApp()
    main_window.show()
    app.exec_()
