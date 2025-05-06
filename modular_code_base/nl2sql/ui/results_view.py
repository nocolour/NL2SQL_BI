import tkinter as tk
from tkinter import ttk, scrolledtext
import logging
import traceback

class ResultsView:
    def __init__(self, parent):
        self.parent = parent
        self.logger = logging.getLogger('nl2sql')
        
        # Create notebook for results
        self.notebook = ttk.Notebook(parent)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Table view tab
        self.table_frame = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(self.table_frame, text="Table View")
        
        # Create treeview for results with scrollbars
        tree_frame = ttk.Frame(self.table_frame)
        tree_frame.pack(fill=tk.BOTH, expand=True)
        
        tree_scroll_y = ttk.Scrollbar(tree_frame)
        tree_scroll_y.pack(side=tk.RIGHT, fill=tk.Y)
        
        tree_scroll_x = ttk.Scrollbar(tree_frame, orient=tk.HORIZONTAL)
        tree_scroll_x.pack(side=tk.BOTTOM, fill=tk.X)
        
        self.result_tree = ttk.Treeview(
            tree_frame, 
            show="headings",
            yscrollcommand=tree_scroll_y.set,
            xscrollcommand=tree_scroll_x.set
        )
        self.result_tree.pack(fill=tk.BOTH, expand=True)
        
        tree_scroll_y.config(command=self.result_tree.yview)
        tree_scroll_x.config(command=self.result_tree.xview)
        
        # Chart view tab
        self.chart_frame = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(self.chart_frame, text="Chart View")
        
        # Summary view tab
        self.summary_frame = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(self.summary_frame, text="Summary")
        
        self.summary_text = scrolledtext.ScrolledText(self.summary_frame, wrap=tk.WORD)
        self.summary_text.pack(fill=tk.BOTH, expand=True)
    
    def display_table(self, df):
        """Display results in the treeview"""
        try:
            # Clear existing data
            for item in self.result_tree.get_children():
                self.result_tree.delete(item)
                
            # Configure columns
            columns = list(df.columns)
            self.result_tree["columns"] = columns
            
            # Configure headings
            self.result_tree["show"] = "headings"
            for col in columns:
                self.result_tree.heading(col, text=col)
                self.result_tree.column(col, width=100)
                
            # Add data rows
            for _, row in df.iterrows():
                values = list(row)
                self.result_tree.insert("", tk.END, values=values)
                
            return True, "Data displayed successfully"
        except Exception as e:
            error_msg = f"Failed to display results: {str(e)}"
            self.logger.error(f"{error_msg}\n{traceback.format_exc()}")
            return False, error_msg
            
    def display_summary(self, summary):
        """Display summary text"""
        try:
            self.summary_text.delete("1.0", tk.END)
            self.summary_text.insert(tk.END, summary)
            return True, "Summary displayed successfully"
        except Exception as e:
            error_msg = f"Failed to display summary: {str(e)}"
            self.logger.error(f"{error_msg}\n{traceback.format_exc()}")
            return False, error_msg