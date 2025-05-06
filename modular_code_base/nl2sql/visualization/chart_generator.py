import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk
import logging
import traceback

class ChartGenerator:
    def __init__(self):
        self.logger = logging.getLogger('nl2sql')
        
    def generate_chart(self, df, chart_frame):
        """Generate appropriate chart for the data"""
        try:
            # Clear current chart
            for widget in chart_frame.winfo_children():
                widget.destroy()

            # Check if we have data and it's suitable for visualization
            if df.empty or len(df.columns) < 2:
                ttk.Label(chart_frame, text="No data available for visualization").pack(expand=True)
                return False, "No data available for visualization"

            # Create figure and axis
            fig, ax = plt.subplots(figsize=(10, 6))

            # Create a default chart based on the data
            if len(df) <= 10:  # Small number of rows - bar chart
                if df.shape[1] == 2:  # Two columns (category and value)
                    x_col = df.columns[0]
                    y_col = df.columns[1]
                    df.plot(kind='bar', x=x_col, y=y_col, ax=ax, legend=False)
                    ax.set_ylabel(y_col)
                else:  # More than two columns - select numeric columns for multi-bar
                    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
                    if numeric_cols:
                        non_numeric_cols = [col for col in df.columns if col not in numeric_cols]
                        if non_numeric_cols:
                            x_col = non_numeric_cols[0]
                            df.plot(kind='bar', x=x_col, y=numeric_cols[:3], ax=ax)
                        else:
                            df.plot(kind='bar', ax=ax)
            else:  # More rows - line chart if there seems to be a trend
                numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
                if df.shape[1] >= 2 and numeric_cols:
                    # Check if first column might be a date or category for x-axis
                    x_col = df.columns[0]
                    y_cols = numeric_cols
                    df.plot(kind='line', x=x_col, y=y_cols[:3], ax=ax, marker='o')

            ax.set_title("Query Results Visualization")
            plt.tight_layout()

            # Embed the plot in the chart frame
            canvas = FigureCanvasTkAgg(fig, master=chart_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
            return True, "Chart generated successfully"

        except Exception as e:
            error_msg = f"Failed to generate chart: {str(e)}"
            self.logger.error(f"{error_msg}\n{traceback.format_exc()}")
            ttk.Label(chart_frame, text=error_msg).pack(expand=True)
            return False, error_msg