# Crawl4AI_v2.py
# This is the main entry point for the Crawl4AI Agent System UI.

import sys
import os
import shutil
import pandas as pd
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QPushButton, QLabel, QFileDialog,
                             QStackedWidget, QLineEdit, QTextEdit, QFrame)
from PyQt5.QtCore import QThread, pyqtSignal, QObject
from PyQt5.QtGui import QFont

# --- Import configurations from existing script ---
from config import CRAWL_CACHE_DIR, KNOWLEDGE_CACHE_DIR, RACE_INPUT_FILE, OUTPUT_DIR

# --- Import main functions from existing scripts ---
from main import main as run_mistral_agent
from gemini import main as run_gemini_processor, DEFAULT_INPUT_FOLDER
from dotenv import load_dotenv


# --- Helper class to redirect stdout/stderr to the UI ---
class Stream(QObject):
    """Redirects console output to a QTextEdit."""
    new_text = pyqtSignal(str)

    def write(self, text):
        self.new_text.emit(str(text))

    def flush(self):
        pass  # Required for stream interface


# --- A QThread worker for running the Mistral Agent ---
class MistralWorker(QThread):
    """Runs the Mistral agent in a separate thread to avoid UI freezing."""

    def __init__(self, env_path, output_path):
        super().__init__()
        self.env_path = env_path
        self.output_path = output_path

    def run(self):
        try:
            if self.env_path and os.path.exists(self.env_path):
                print(f"Loading environment variables from: {self.env_path}")
                load_dotenv(dotenv_path=self.env_path, override=True)
            else:
                print("Warning: .env path not specified or not found. Using default environment.")

            run_mistral_agent(output_dir_override=self.output_path)
        except Exception as e:
            print(f"An error occurred in the Mistral Agent: {e}")


# --- A QThread worker for running the Gemini Processor ---
class GeminiWorker(QThread):
    """Runs the Gemini processor in a separate thread."""

    def __init__(self, env_path, output_path, input_path):
        super().__init__()
        self.env_path = env_path
        self.output_path = output_path
        self.input_path = input_path

    def run(self):
        try:
            if self.env_path and os.path.exists(self.env_path):
                print(f"Loading environment variables from: {self.env_path}")
                load_dotenv(dotenv_path=self.env_path, override=True)
            else:
                print("Warning: .env path not specified or not found. Using default environment.")

            run_gemini_processor(output_dir_override=self.output_path, input_dir_override=self.input_path)
        except Exception as e:
            print(f"An error occurred in the Gemini Processor: {e}")


class Crawl4AIApp(QMainWindow):
    """The main application window."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Crawl4AI Agent System")
        self.setGeometry(100, 100, 900, 700)
        self.setMinimumSize(800, 600)

        self.stacked_widget = QStackedWidget()
        self.setCentralWidget(self.stacked_widget)

        self.home_screen = HomeScreen()
        self.mistral_screen = MistralScreen()
        self.gemini_screen = GeminiScreen()

        self.stacked_widget.addWidget(self.home_screen)
        self.stacked_widget.addWidget(self.mistral_screen)
        self.stacked_widget.addWidget(self.gemini_screen)

        self.home_screen.mistral_selected.connect(lambda: self.stacked_widget.setCurrentWidget(self.mistral_screen))
        self.home_screen.gemini_selected.connect(lambda: self.stacked_widget.setCurrentWidget(self.gemini_screen))
        self.mistral_screen.back_to_home.connect(self.go_to_home)
        self.gemini_screen.back_to_home.connect(self.go_to_home)

    def go_to_home(self):
        self.stacked_widget.setCurrentWidget(self.home_screen)


class HomeScreen(QWidget):
    """The initial model selection screen."""
    mistral_selected = pyqtSignal()
    gemini_selected = pyqtSignal()

    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(50, 50, 50, 50)
        layout.setSpacing(20)

        title = QLabel("Crawl4AI Agent System")
        title.setObjectName("TitleLabel")

        subtitle = QLabel("Please choose which model you would like to use.")
        subtitle.setObjectName("SubtitleLabel")

        self.mistral_button = QPushButton("Mistral Web Analyst")
        self.mistral_button.setObjectName("PrimaryButton")
        self.mistral_button.setFixedHeight(60)
        self.mistral_button.clicked.connect(self.mistral_selected.emit)

        self.gemini_button = QPushButton("Gemini Image Processor")
        self.gemini_button.setObjectName("SecondaryButton")
        self.gemini_button.setFixedHeight(60)
        self.gemini_button.clicked.connect(self.gemini_selected.emit)

        layout.addStretch(1)
        layout.addWidget(title)
        layout.addWidget(subtitle)
        layout.addSpacing(40)
        layout.addWidget(self.mistral_button)
        layout.addWidget(self.gemini_button)
        layout.addStretch(2)


class BaseAgentScreen(QWidget):
    """A base class for consistent layout on agent screens."""
    back_to_home = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.worker = None
        self.env_file_path = QLineEdit()
        self.output_dir_path = QLineEdit()

        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(20, 10, 20, 20)
        self.main_layout.setSpacing(15)

        header_layout = QHBoxLayout()
        back_button = QPushButton("← Back to Home")
        back_button.setObjectName("BackButton")
        back_button.setFixedWidth(150)
        back_button.clicked.connect(self.back_to_home.emit)
        header_layout.addWidget(back_button)
        header_layout.addStretch()
        self.main_layout.addLayout(header_layout)

    def _create_group_box(self, title):
        frame = QFrame()
        frame.setObjectName("GroupBox")
        layout = QVBoxLayout(frame)

        label = QLabel(title)
        label.setObjectName("GroupLabel")
        layout.addWidget(label)

        return frame, layout

    def _create_log_box(self):
        log_frame, log_layout = self._create_group_box("Live Log")
        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)
        log_layout.addWidget(self.log_output)
        return log_frame

    def browse_env_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select .env File", "", "Environment Files (*.env)")
        if file_path:
            self.env_file_path.setText(file_path)

    def browse_output_dir(self):
        dir_path = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if dir_path:
            self.output_dir_path.setText(dir_path)


class MistralScreen(BaseAgentScreen):
    """The UI for the Mistral Web Analyst."""

    def __init__(self):
        super().__init__()

        inputs_frame, inputs_layout = self._create_group_box("1. Select Inputs")

        # .env File Path
        env_file_layout = QHBoxLayout()
        self.env_file_path.setPlaceholderText(".env File Path (for API keys)")
        browse_env_button = QPushButton("Browse...")
        browse_env_button.clicked.connect(self.browse_env_file)
        env_file_layout.addWidget(self.env_file_path)
        env_file_layout.addWidget(browse_env_button)
        inputs_layout.addLayout(env_file_layout)

        # Input Data File
        input_file_layout = QHBoxLayout()
        self.input_file_path = QLineEdit()
        self.input_file_path.setPlaceholderText("Input Data File (JSON/CSV/XLSX)")
        browse_input_button = QPushButton("Browse...")
        browse_input_button.clicked.connect(self.browse_input_file)
        input_file_layout.addWidget(self.input_file_path)
        input_file_layout.addWidget(browse_input_button)
        inputs_layout.addLayout(input_file_layout)

        # Output Directory
        output_dir_layout = QHBoxLayout()
        self.output_dir_path.setPlaceholderText(f"Output Directory (default: {os.path.abspath(OUTPUT_DIR)})")
        browse_output_button = QPushButton("Browse...")
        browse_output_button.clicked.connect(self.browse_output_dir)
        output_dir_layout.addWidget(self.output_dir_path)
        output_dir_layout.addWidget(browse_output_button)
        inputs_layout.addLayout(output_dir_layout)

        self.main_layout.addWidget(inputs_frame)

        run_frame, run_layout = self._create_group_box("2. Run Analysis")
        self.start_button = QPushButton("Start Processing")
        self.start_button.setObjectName("StartButton")
        self.start_button.setFixedHeight(45)
        self.start_button.clicked.connect(self.start_processing)
        run_layout.addWidget(self.start_button)
        self.main_layout.addWidget(run_frame)

        utilities_frame, utilities_layout = self._create_group_box("Utilities")
        buttons_layout = QHBoxLayout()
        clear_crawl_cache_button = QPushButton("Clear Crawl Cache")
        clear_knowledge_cache_button = QPushButton("Clear Knowledge Cache")
        clear_crawl_cache_button.clicked.connect(lambda: self.clear_directory(CRAWL_CACHE_DIR, "Crawl Cache"))
        clear_knowledge_cache_button.clicked.connect(
            lambda: self.clear_directory(KNOWLEDGE_CACHE_DIR, "Knowledge Cache"))
        buttons_layout.addWidget(clear_crawl_cache_button)
        buttons_layout.addWidget(clear_knowledge_cache_button)
        utilities_layout.addLayout(buttons_layout)
        self.main_layout.addWidget(utilities_frame)

        self.main_layout.addWidget(self._create_log_box())
        self.main_layout.setStretch(4, 1)

    def browse_input_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Input File", "", "Data Files (*.json *.csv *.xlsx)")
        if file_path:
            self.input_file_path.setText(file_path)

    def clear_directory(self, dir_path, dir_name):
        self.log_output.append(f"Attempting to clear {dir_name} at '{dir_path}'...")
        try:
            if os.path.exists(dir_path):
                shutil.rmtree(dir_path)
                os.makedirs(dir_path)
                self.log_output.append(f"✅ Success: Cleared and recreated '{dir_path}'.")
            else:
                os.makedirs(dir_path)
                self.log_output.append(f"✅ Success: Directory did not exist. Created '{dir_path}'.")
        except Exception as e:
            self.log_output.append(f"❌ Error clearing {dir_name}: {e}")

    def start_processing(self):
        input_path = self.input_file_path.text()
        if not input_path or not os.path.exists(input_path):
            self.log_output.append("❌ Error: Please select a valid input data file.")
            return

        try:
            if input_path.lower().endswith('.csv'):
                self.log_output.append("CSV file detected. Converting to JSON...")
                df = pd.read_csv(input_path)
                df.to_json(RACE_INPUT_FILE, orient='records', indent=4)
                self.log_output.append(f"✅ Successfully converted CSV to '{RACE_INPUT_FILE}'.")
            elif input_path.lower().endswith('.json'):
                self.log_output.append("JSON file detected. Copying to target...")
                shutil.copy(input_path, RACE_INPUT_FILE)
                self.log_output.append(f"✅ Copied input file to '{RACE_INPUT_FILE}'.")
            else:
                self.log_output.append("❌ Error: Unsupported file type. Please use JSON or CSV.")
                return
        except Exception as e:
            self.log_output.append(f"❌ Error during file preparation: {e}")
            return

        self.start_button.setText("Processing...")
        self.start_button.setEnabled(False)
        self.log_output.clear()

        output_path = self.output_dir_path.text() or None
        self.worker = MistralWorker(env_path=self.env_file_path.text(), output_path=output_path)
        self.worker.start()
        self.worker.finished.connect(lambda: (
            self.start_button.setText("Start Processing"),
            self.start_button.setEnabled(True)
        ))


class GeminiScreen(BaseAgentScreen):
    """The UI for the Gemini Image Processor."""

    def __init__(self):
        super().__init__()

        config_frame, config_layout = self._create_group_box("1. Configuration")

        # .env File Path
        env_file_layout = QHBoxLayout()
        self.env_file_path.setPlaceholderText(".env File Path (for GEMINI_API_KEY)")
        browse_env_button = QPushButton("Browse...")
        browse_env_button.clicked.connect(self.browse_env_file)
        env_file_layout.addWidget(self.env_file_path)
        env_file_layout.addWidget(browse_env_button)
        config_layout.addLayout(env_file_layout)

        # Image Input Directory
        input_dir_layout = QHBoxLayout()
        self.input_dir_path = QLineEdit()
        self.input_dir_path.setPlaceholderText(f"Image Input Directory (default: '{DEFAULT_INPUT_FOLDER}')")
        browse_input_button = QPushButton("Browse...")
        browse_input_button.clicked.connect(self.browse_input_dir)
        input_dir_layout.addWidget(self.input_dir_path)
        input_dir_layout.addWidget(browse_input_button)
        config_layout.addLayout(input_dir_layout)

        # Output Directory
        output_dir_layout = QHBoxLayout()
        self.output_dir_path.setPlaceholderText("Output Directory (default: same folder as this app)")
        browse_output_button = QPushButton("Browse...")
        browse_output_button.clicked.connect(self.browse_output_dir)
        output_dir_layout.addWidget(self.output_dir_path)
        output_dir_layout.addWidget(browse_output_button)
        config_layout.addLayout(output_dir_layout)

        info_label = QLabel(
            "This tool processes all images (.png, .jpg, .jpeg) found in the specified input directory."
        )
        info_label.setObjectName("InfoLabel")
        info_label.setWordWrap(True)
        config_layout.addSpacing(10)
        config_layout.addWidget(info_label)
        self.main_layout.addWidget(config_frame)

        run_frame, run_layout = self._create_group_box("2. Run Processor")
        self.start_button = QPushButton("Start Processing Images")
        self.start_button.setObjectName("StartButton")
        self.start_button.setFixedHeight(45)
        self.start_button.clicked.connect(self.start_processing)
        run_layout.addWidget(self.start_button)
        self.main_layout.addWidget(run_frame)
        self.main_layout.addStretch(1)

        self.main_layout.addWidget(self._create_log_box())
        self.main_layout.setStretch(3, 2)

    def browse_input_dir(self):
        dir_path = QFileDialog.getExistingDirectory(self, "Select Image Input Directory")
        if dir_path:
            self.input_dir_path.setText(dir_path)

    def start_processing(self):
        self.start_button.setText("Processing...")
        self.start_button.setEnabled(False)
        self.log_output.clear()

        output_path = self.output_dir_path.text() or None
        input_path = self.input_dir_path.text() or None
        self.worker = GeminiWorker(env_path=self.env_file_path.text(), output_path=output_path, input_path=input_path)
        self.worker.start()
        self.worker.finished.connect(lambda: (
            self.start_button.setText("Start Processing Images"),
            self.start_button.setEnabled(True)
        ))


def main_ui():
    """Initializes and runs the PyQt application."""
    app = QApplication(sys.argv)

    app.setStyleSheet("""
        QWidget {
            font-family: 'Segoe UI', Arial, sans-serif;
            font-size: 10pt;
            color: #e0e0e0;
        }
        QMainWindow, QWidget {
            background-color: #2c3e50;
        }
        #TitleLabel {
            font-size: 28pt;
            font-weight: bold;
            color: #ffffff;
        }
        #SubtitleLabel {
            font-size: 12pt;
            color: #bdc3c7;
        }
        #PrimaryButton {
            font-size: 14pt;
            font-weight: bold;
            background-color: #3498db;
            color: white;
            border: none;
            padding: 10px;
            border-radius: 5px;
        }
        #PrimaryButton:hover {
            background-color: #2980b9;
        }
        #SecondaryButton {
            font-size: 14pt;
            background-color: #95a5a6;
            color: #2c3e50;
            border: none;
            padding: 10px;
            border-radius: 5px;
        }
        #SecondaryButton:hover {
            background-color: #7f8c8d;
        }
        #BackButton {
            background-color: transparent;
            border: 1px solid #7f8c8d;
            border-radius: 5px;
            padding: 8px;
        }
        #BackButton:hover {
            background-color: #34495e;
        }
        #GroupBox {
            background-color: #34495e;
            border-radius: 8px;
        }
        #GroupLabel {
            font-size: 11pt;
            font-weight: bold;
            color: #ecf0f1;
            padding-bottom: 5px;
        }
        #InfoLabel {
            background-color: #2c3e50;
            border: 1px solid #4a627a;
            padding: 10px;
            border-radius: 5px;
            color: #bdc3c7;
        }
        QPushButton {
            background-color: #5d6d7e;
            border: none;
            padding: 8px 12px;
            border-radius: 4px;
        }
        QPushButton:hover {
            background-color: #718093;
        }
        QPushButton:disabled {
            background-color: #4a5a6a;
            color: #95a5a6;
        }
        #StartButton {
            background-color: #27ae60;
            color: white;
            font-size: 12pt;
            font-weight: bold;
        }
        #StartButton:hover {
            background-color: #229954;
        }
        QLineEdit {
            background-color: #2c3e50;
            border: 1px solid #5d6d7e;
            padding: 8px;
            border-radius: 4px;
            color: #ecf0f1;
        }
        QTextEdit {
            font-family: 'Consolas', 'Monaco', monospace;
            background-color: #212f3d;
            color: #f2f2f2;
            border: 1px solid #4a627a;
            border-radius: 4px;
            padding: 5px;
        }
    """)

    window = Crawl4AIApp()

    sys.stdout = Stream(new_text=lambda text: (
        window.mistral_screen.log_output.moveCursor(window.mistral_screen.log_output.textCursor().End),
        window.mistral_screen.log_output.insertPlainText(text),
        window.gemini_screen.log_output.moveCursor(window.gemini_screen.log_output.textCursor().End),
        window.gemini_screen.log_output.insertPlainText(text)
    ))
    sys.stderr = sys.stdout

    window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    os.makedirs(CRAWL_CACHE_DIR, exist_ok=True)
    os.makedirs(KNOWLEDGE_CACHE_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(DEFAULT_INPUT_FOLDER, exist_ok=True)

    main_ui()