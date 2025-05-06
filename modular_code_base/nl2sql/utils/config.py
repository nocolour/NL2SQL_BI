import json
import os
import logging
import traceback

class Config:
    def __init__(self):
        self.logger = logging.getLogger('nl2sql')
        
        # Default configuration
        self.db_config = {
            "host": "localhost",
            "user": "root",
            "password": "",
            "database": "",
            "port": 3306
        }
        
        self.openai_api_key = ""
        self.ai_model = "gpt-4o-mini"
        
        # Example queries
        self.example_queries = [
            "Show all customers from the USA",
            "What are the top 5 products by sales?",
            "List all employees hired in 2022",
            "Show me the total revenue by month",
            "Which customers have placed more than 10 orders?",
            "Show all tables in the database",
            "Display the schema for the customers table"
        ]
    
    def get_config_path(self, custom_path=None):
        """Get the path to the configuration file"""
        if custom_path:
            return custom_path
        return os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "nl2sql_config.json")
        
    def load(self, config_path=None):
        """Load configuration from config file"""
        config_file = self.get_config_path(config_path)
        
        if os.path.exists(config_file):
            try:
                with open(config_file, "r") as f:
                    config = json.load(f)
                
                if "database" in config:
                    self.db_config = config["database"]
                
                if "openai_api_key" in config:
                    self.openai_api_key = config["openai_api_key"]
                
                if "ai_model" in config:
                    self.ai_model = config["ai_model"]
                    
                if "example_queries" in config:
                    self.example_queries = config["example_queries"]
                    
                return True, ""
            except Exception as e:
                error_msg = f"Failed to load configuration: {str(e)}"
                self.logger.error(f"{error_msg}\n{traceback.format_exc()}")
                return False, error_msg
        
        return True, "No configuration file found, using defaults."
    
    def save(self, config_path=None):
        """Save configuration to config file"""
        config_file = self.get_config_path(config_path)
        
        try:
            config = {
                "database": self.db_config,
                "openai_api_key": self.openai_api_key,
                "ai_model": self.ai_model,
                "example_queries": self.example_queries
            }
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(config_file), exist_ok=True)
            
            with open(config_file, "w") as f:
                json.dump(config, f, indent=4)
                
            return True, "Configuration saved successfully."
        except Exception as e:
            error_msg = f"Failed to save configuration: {str(e)}"
            self.logger.error(f"{error_msg}\n{traceback.format_exc()}")
            return False, error_msg
    
    def get_example_queries(self):
        """Get the list of example queries"""
        return self.example_queries
    
    def to_dict(self):
        """Convert the configuration to a dictionary"""
        return {
            "database": self.db_config,
            "openai_api_key": self.openai_api_key,
            "ai_model": self.ai_model,
            "example_queries": self.example_queries
        }