from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtWidgets import QWidget, QVBoxLayout

def clear_window(widget: QWidget):
    """
    Clears all widgets from the given QWidget.
    """
    for i in reversed(range(widget.layout().count())):
        child = widget.layout().itemAt(i).widget()
        if child:
            child.setParent(None)


def display_plot(fig, parent_widget: QWidget):
    """
    Embeds a Matplotlib figure into a PyQt5 widget.

    Args:
        fig: Matplotlib figure to be displayed.
        parent_widget: The QWidget where the figure should be displayed.
    """
    # Clear the current layout
    clear_window(parent_widget)

    # Create a canvas to embed the Matplotlib figure
    canvas = FigureCanvas(fig)

    # Create a layout for the parent widget if it doesn't exist
    if not parent_widget.layout():
        layout = QVBoxLayout()
        parent_widget.setLayout(layout)

    # Add the canvas to the layout
    parent_widget.layout().addWidget(canvas)

    # Draw the canvas
    canvas.draw()
