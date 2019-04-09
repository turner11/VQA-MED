import matplotlib.pyplot as plt

import PyQt5.QtWidgets as qt
import sys
from matplotlib.backends.backend_qt5 import FigureCanvasQT

class mainWindow(qt.QTabWidget):
    def __init__(self, figures, parent=None):
        super().__init__(parent)

        # GUI configuration
        self.figures = figures
        self.tabs = []
        plt.ion()
        for i, figure  in enumerate(figures):
            tab = qt.QWidget()
            self.tabs.append(tab)
            self.addTab(tab , f"Tab {i}")
            layout = qt.QVBoxLayout()
            tab.setLayout(layout)

            # self.resize(800, 480)
            try:

                canvas = FigureCanvasQT(figure)
                layout.addWidget(canvas,1)

                # canvas.draw_idle()
                canvas.draw()
            except Exception as ex:
                pass
                raise
        plt.ion()




def main():
    import numpy as np
    import matplotlib.pyplot as plt

    x = np.linspace(1, 2 * np.pi, 100)
    figures = []
    for i in range(1, 3):
        fig, ax = plt.subplots()
        y = np.sin(np.pi * i * x) + 0.1 * np.random.randn(100)
        ax.plot(x, y)
        figures.append(fig)

    open_window(figures)

def open_window(figures):
    app = qt.QApplication(sys.argv)
    main = mainWindow(figures)
    main.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()

