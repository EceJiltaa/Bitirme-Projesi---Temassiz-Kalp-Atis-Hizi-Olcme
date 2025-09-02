import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout

class PlotWidget(QWidget):
    def __init__(self, title):
        super().__init__()

        self.figure = plt.figure(figsize=(6, 4))
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)
        self.ax.set_title(title)
        self.ax.set_xlabel('Frames')
        self.ax.set_ylabel('Red Change')
        self.lines, = self.ax.plot([], [], 'b-')
        self.x_data = []
        self.y_data = []

        layout = QHBoxLayout()  # Yatay düzen kullanılıyor
        layout.addWidget(self.canvas)

        # Yeni bir widget oluşturup, diğer widget'ları içine yerleştiriyoruz
        widget = QWidget()
        widget.setLayout(layout)

        # Ana düzeneğe bu widget'ı ekliyoruz
        main_layout = QVBoxLayout()
        main_layout.addWidget(widget)
        self.setLayout(main_layout)

    def update_plot(self, y):
        self.y_data.append(y)
        self.x_data.append(len(self.y_data))
        self.lines.set_data(self.x_data, self.y_data)
        self.ax.relim()
        self.ax.autoscale_view()
        self.canvas.draw()
