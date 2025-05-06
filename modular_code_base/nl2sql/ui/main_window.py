import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import logging
import traceback
from datetime import datetime
from nl2sql.ui.settings_dialog import SettingsDialog
from nl2sql.ui.results_view import ResultsView

class MainWindow:
    def __init__(self, root, config, db_connector, sql_generator, chart_generator, sql_validator):
        self.root = root
        self.config = config
        self.db_connector = db_connector
        self.sql_generator = sql_generator
        self.chart_generator = chart_generator
        self.sql_validator = sql_validator
        self.logger = logging.getLogger('nl2sql')
        
        # Set up main window
        self.root.title("Natural Language to SQL Query System")
        self.root.geometry("1200x800")
        self.root.minsize(800, 600)
        
        # Create the UI components
        self.create_menu()
        self.create_ui()
        
    def create_menu(self):
        """Create application menu"""
        menubar = tk.Menu(self.root)
        
        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="Settings", command=self.show_settings)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)
        menubar.add_cascade(label="File", menu=file_menu)
        
        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        help_menu.add_command(label="About", command=self.show_about)
        menubar.add_cascade(label="Help", menu=help_menu)
        
        self.root.config(menu=menubar)
        
    def create_ui(self):
        """Create the main user interface"""
        # Create main frame
        main_frame = ttk.Frame(self.root, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create top frame for query input
        query_frame = ttk.LabelFrame(main_frame, text="Natural Language Query", padding=10)
        query_frame.pack(fill=tk.X, pady=5)
        
        # Query text input
        self.query_text = scrolledtext.ScrolledText(query_frame, height=4, wrap=tk.WORD)
        self.query_text.pack(fill=tk.X, expand=True, pady=5)
        
        # Example queries dropdown
        example_frame = ttk.Frame(query_frame)
        example_frame.pack(fill=tk.X, expand=True, pady=5)
        
        ttk.Label(example_frame, text="Example queries:").pack(side=tk.LEFT, padx=5)
        self.example_var = tk.StringVar()
        example_combo = ttk.Combobox(
            example_frame, 
            textvariable=self.example_var, 
            width=50, 
            values=self.config.example_queries
        )
        example_combo.pack(side=tk.LEFT, padx=5)
        example_combo.bind("<<ComboboxSelected>>", self.use_example)
        
        # Buttons frame
        button_frame = ttk.Frame(query_frame)
        button_frame.pack(fill=tk.X, expand=True, pady=5)
        
        ttk.Button(button_frame, text="Execute Query", command=self.execute_query).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Clear", command=self.clear_query).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="View Schema", command=self.view_schema).pack(side=tk.LEFT, padx=5)
        
        # Create paned window for SQL/results
        paned = ttk.PanedWindow(main_frame, orient=tk.VERTICAL)
        paned.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # SQL frame
        sql_frame = ttk.LabelFrame(paned, text="Generated SQL", padding=10)
        paned.add(sql_frame, weight=1)
        
        self.sql_text = scrolledtext.ScrolledText(sql_frame, height=4, wrap=tk.WORD)
        self.sql_text.pack(fill=tk.BOTH, expand=True)
        
        # Results frame with notebook for table/chart views
        results_frame = ttk.LabelFrame(paned, text="Query Results", padding=10)
        paned.add(results_frame, weight=3)
        
        # Add results view
        self.results_view = ResultsView(results_frame)
        
        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        status_bar = ttk.Label(main_frame, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(fill=tk.X, pady=5)
        
    def show_settings(self):
        """Show settings dialog"""
        settings_dialog = SettingsDialog(self.root, self.config, self.db_connector)
        settings_dialog.show()
        
        # Update SQL generator with new settings
        self.sql_generator.api_key = self.config.openai_api_key
        self.sql_generator.model_name = self.config.ai_model
        
    def execute_query(self):
        """Process the natural language query and execute SQL"""
        try:
            # Get the query text
            query = self.query_text.get("1.0", tk.END).strip()
            
            if not query:
                messagebox.showwarning("Input Error", "Please enter a query.")
                return
                
            if not self.config.openai_api_key:
                messagebox.showwarning("API Key Required", "Please set your OpenAI API key in settings.")
                return
                
            # Update status
            self.status_var.set("Generating SQL query...")
            self.root.update_idletasks()
            
            # Get database schema
            success, schema_info = self.db_connector.get_schema()
            if not success:
                messagebox.showerror("Schema Error", f"Failed to get database schema: {schema_info}")
                self.status_var.set("Error occurred")
                return
                
            # Generate SQL using the selected model
            success, sql_query = self.sql_generator.generate_sql(query, schema_info)
            if not success:
                messagebox.showerror("SQL Generation Error", sql_query)
                self.status_var.set("Error occurred")
                return
                
            # Fix ambiguous column references in the query
            sql_query = self.db_connector.fix_ambiguous_columns(sql_query)
            
            # Display the SQL
            self.sql_text.delete("1.0", tk.END)
            self.sql_text.insert(tk.END, sql_query)
            
            # Validate SQL
            valid, error_msg = self.sql_validator.validate(sql_query)
            if not valid:
                messagebox.showerror("SQL Validation Error", error_msg)
                self.status_var.set("Error occurred")
                return
                
            # Update status
            self.status_var.set("Executing query...")
            self.root.update_idletasks()
            
            # Execute the query
            success, result = self.db_connector.execute_query(sql_query)
            if not success:
                messagebox.showerror("Query Execution Error", result)
                self.status_var.set("Error occurred")
                return
                
            df = result
                
            # Display results
            success, message = self.results_view.display_table(df)
            if not success:
                messagebox.showwarning("Display Warning", message)
                
            # Generate chart
            self.chart_generator.generate_chart(df, self.results_view.chart_frame)
            
            # Generate summary
            success, summary = self.sql_generator.generate_summary(query, sql_query, df)
            if success:
                self.results_view.display_summary(summary)
            else:
                self.results_view.display_summary(f"Could not generate summary: {summary}")
                
            # Update status
            self.status_var.set(f"Query completed: {len(df)} rows returned")
            
        except Exception as e:
            error_msg = f"An error occurred: {str(e)}"
            today = datetime.now().strftime("%Y-%m-%d")
            log_file = f"error_{today}.log"
            self.logger.error(f"Query execution error: {error_msg}\n{traceback.format_exc()}")
            messagebox.showerror("Error", f"{error_msg}\nError details exported to {log_file}")
            self.status_var.set("Error occurred")
            
    def use_example(self, event):
        """Fill the query text box with the selected example"""
        try:
            example = self.example_var.get()
            self.query_text.delete("1.0", tk.END)
            self.query_text.insert(tk.END, example)
        except Exception as e:
            error_msg = f"Failed to use example: {str(e)}"
            self.logger.error(f"{error_msg}\n{traceback.format_exc()}")
            messagebox.showerror("Error", error_msg)
            
    def clear_query(self):
        """Clear the query text box"""
        try:
            self.query_text.delete("1.0", tk.END)
        except Exception as e:
            error_msg = f"Failed to clear query: {str(e)}"
            self.logger.error(f"{error_msg}\n{traceback.format_exc()}")
            messagebox.showerror("Error", error_msg)
            
    def view_schema(self):
        """View the database schema"""
        try:
            success, schema_info = self.db_connector.get_schema()
            
            if not success:
                messagebox.showerror("Schema Error", schema_info)
                return
                
            # Show in a new window
            schema_window = tk.Toplevel(self.root)
            schema_window.title("Database Schema")
            schema_window.geometry("600x400")
            schema_window.transient(self.root)
            
            schema_text = scrolledtext.ScrolledText(schema_window, wrap=tk.WORD)
            schema_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
            schema_text.insert(tk.END, schema_info)
            schema_text.config(state=tk.DISABLED)
            
        except Exception as e:
            error_msg = f"Failed to view schema: {str(e)}"
            self.logger.error(f"{error_msg}\n{traceback.format_exc()}")
            messagebox.showerror("Error", error_msg)
            
    def show_about(self):
        """Show about dialog"""
        try:
            messagebox.showinfo(
                "About NL2SQL Query System",
                "Natural Language to SQL Query System\n\n"
                "This application allows you to query MySQL databases "
                "using natural language. It converts your questions into "
                "SQL queries using OpenAI's models.\n\n"
                "Configure your database connection, API key, and AI model in the "
                "settings to get started."
            )
        except Exception as e:
            error_msg = f"Failed to show about dialog: {str(e)}"
            self.logger.error(f"{error_msg}\n{traceback.format_exc()}")