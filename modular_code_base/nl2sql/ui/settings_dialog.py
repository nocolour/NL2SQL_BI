import tkinter as tk
from tkinter import ttk, messagebox
import logging
import traceback

class SettingsDialog:
    def __init__(self, parent, config, db_connector):
        self.parent = parent
        self.config = config
        self.db_connector = db_connector
        self.logger = logging.getLogger('nl2sql')
        
    def show(self):
        """Show settings dialog"""
        try:
            settings_window = tk.Toplevel(self.parent)
            settings_window.title("Settings")
            settings_window.geometry("500x400")
            settings_window.resizable(False, False)
            settings_window.transient(self.parent)
            settings_window.grab_set()

            # Create notebook for settings categories
            notebook = ttk.Notebook(settings_window)
            notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

            # Database settings tab
            db_frame = ttk.Frame(notebook, padding=10)
            notebook.add(db_frame, text="Database")

            # Grid layout for database settings
            ttk.Label(db_frame, text="Host:").grid(row=0, column=0, sticky=tk.W, pady=5)
            ttk.Label(db_frame, text="User:").grid(row=1, column=0, sticky=tk.W, pady=5)
            ttk.Label(db_frame, text="Password:").grid(row=2, column=0, sticky=tk.W, pady=5)
            ttk.Label(db_frame, text="Database:").grid(row=3, column=0, sticky=tk.W, pady=5)
            ttk.Label(db_frame, text="Port:").grid(row=4, column=0, sticky=tk.W, pady=5)

            # Entry fields with current values
            host_var = tk.StringVar(value=self.config.db_config.get("host", ""))
            user_var = tk.StringVar(value=self.config.db_config.get("user", ""))
            password_var = tk.StringVar(value=self.config.db_config.get("password", ""))
            database_var = tk.StringVar(value=self.config.db_config.get("database", ""))
            port_var = tk.StringVar(value=str(self.config.db_config.get("port", 3306)))

            host_entry = ttk.Entry(db_frame, textvariable=host_var, width=30)
            host_entry.grid(row=0, column=1, sticky=tk.W, pady=5)

            user_entry = ttk.Entry(db_frame, textvariable=user_var, width=30)
            user_entry.grid(row=1, column=1, sticky=tk.W, pady=5)

            password_entry = ttk.Entry(db_frame, textvariable=password_var, width=30, show="*")
            password_entry.grid(row=2, column=1, sticky=tk.W, pady=5)

            database_entry = ttk.Entry(db_frame, textvariable=database_var, width=30)
            database_entry.grid(row=3, column=1, sticky=tk.W, pady=5)

            port_entry = ttk.Entry(db_frame, textvariable=port_var, width=30)
            port_entry.grid(row=4, column=1, sticky=tk.W, pady=5)

            # Test connection button
            test_conn_btn = ttk.Button(
                db_frame, 
                text="Test Connection",
                command=lambda: self._test_connection(
                    host_var.get(), user_var.get(),
                    password_var.get(), database_var.get(),
                    port_var.get()
                )
            )
            test_conn_btn.grid(row=5, column=0, columnspan=2, pady=10)

            # API settings tab
            api_frame = ttk.Frame(notebook, padding=10)
            notebook.add(api_frame, text="API Settings")

            ttk.Label(api_frame, text="OpenAI API Key:").grid(row=0, column=0, sticky=tk.W, pady=5)
            ttk.Label(api_frame, text="AI Model:").grid(row=1, column=0, sticky=tk.W, pady=5)

            api_key_var = tk.StringVar(value=self.config.openai_api_key)
            model_var = tk.StringVar(value=self.config.ai_model)

            api_key_entry = ttk.Entry(api_frame, textvariable=api_key_var, width=40)
            api_key_entry.grid(row=0, column=1, sticky=tk.W, pady=5)

            # Add model selection dropdown
            model_combo = ttk.Combobox(
                api_frame, 
                textvariable=model_var, 
                width=30, 
                values=["gpt-4o-mini", "gpt-4.1-mini", "gpt-4o", "gpt-3.5-turbo"]
            )
            model_combo.grid(row=1, column=1, sticky=tk.W, pady=5)
            model_combo.state(["readonly"])  # Make it read-only

            # Buttons frame
            btn_frame = ttk.Frame(settings_window)
            btn_frame.pack(fill=tk.X, padx=10, pady=10)

            # Save and Cancel buttons
            ttk.Button(
                btn_frame, 
                text="Save",
                command=lambda: self._save_settings(
                    host_var.get(), user_var.get(),
                    password_var.get(), database_var.get(),
                    port_var.get(), api_key_var.get(),
                    model_var.get(), settings_window
                )
            ).pack(side=tk.RIGHT, padx=5)

            ttk.Button(
                btn_frame, 
                text="Cancel",
                command=settings_window.destroy
            ).pack(side=tk.RIGHT, padx=5)
            
            return settings_window
            
        except Exception as e:
            error_msg = f"Failed to show settings dialog: {str(e)}"
            self.logger.error(f"{error_msg}\n{traceback.format_exc()}")
            return None
    
    def _test_connection(self, host, user, password, database, port):
        """Test database connection with provided credentials"""
        # Create temporary config
        temp_config = {
            "host": host,
            "user": user,
            "password": password,
            "database": database,
            "port": int(port)
        }
        
        # Save original config
        original_config = self.db_connector.db_config
        
        try:
            # Set temporary config
            self.db_connector.db_config = temp_config
            
            # Test connection
            success, message = self.db_connector.test_connection()
            
            if success:
                messagebox.showinfo("Connection Test", message)
            else:
                messagebox.showerror("Connection Error", message)
                
        except Exception as e:
            error_msg = f"Failed to test connection: {str(e)}"
            self.logger.error(f"{error_msg}\n{traceback.format_exc()}")
            messagebox.showerror("Error", error_msg)
            
        finally:
            # Restore original config
            self.db_connector.db_config = original_config
    
    def _save_settings(self, host, user, password, database, port, api_key, model, window):
        """Save the settings and close the settings window"""
        try:
            self.config.db_config = {
                "host": host,
                "user": user,
                "password": password,
                "database": database,
                "port": int(port)
            }

            self.config.openai_api_key = api_key
            self.config.ai_model = model

            # Save to config file
            success, message = self.config.save()
            
            if success:
                window.destroy()
            else:
                messagebox.showerror("Settings Error", message)
                
        except Exception as e:
            error_msg = f"Failed to save settings: {str(e)}"
            self.logger.error(f"{error_msg}\n{traceback.format_exc()}")
            messagebox.showerror("Settings Error", error_msg)