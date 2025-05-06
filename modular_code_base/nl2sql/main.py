import tkinter as tk
import sys
import os

# Import modules
from nl2sql.utils.logger import setup_logger
from nl2sql.utils.config import Config
from nl2sql.utils.sql_validator import SQLValidator
from nl2sql.database.connector import DatabaseConnector
from nl2sql.nlp.sql_generator import SQLGenerator
from nl2sql.visualization.chart_generator import ChartGenerator
from nl2sql.ui.main_window import MainWindow

def main():
    # Setup logger
    logger = setup_logger()
    
    # Create root window
    root = tk.Tk()
    
    try:
        # Load configuration
        config = Config()
        success, message = config.load()
        if not success:
            tk.messagebox.showwarning("Configuration Warning", message)
        
        # Create components
        db_connector = DatabaseConnector(config.db_config)
        sql_generator = SQLGenerator(config.openai_api_key, config.ai_model)
        chart_generator = ChartGenerator()
        sql_validator = SQLValidator()
        
        # Create main window
        app = MainWindow(root, config, db_connector, sql_generator, chart_generator, sql_validator)
        
        # Start the application
        root.mainloop()
    except Exception as e:
        logger.error(f"Application error: {str(e)}", exc_info=True)
        import traceback
        tk.messagebox.showerror("Error", f"An error occurred starting the application: {str(e)}")
        root.destroy()

if __name__ == "__main__":
    main()