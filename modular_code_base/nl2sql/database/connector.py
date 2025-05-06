import mysql.connector
import pandas as pd
import re
import logging
import traceback

class DatabaseConnector:
    def __init__(self, db_config):
        self.db_config = db_config
        self.logger = logging.getLogger('nl2sql')
        
    def test_connection(self):
        """Test the database connection"""
        try:
            connection = mysql.connector.connect(**self.db_config)
            connection.close()
            return True, "Database connection successful!"
        except Exception as e:
            error_msg = f"Failed to connect to database: {str(e)}"
            self.logger.error(f"{error_msg}\n{traceback.format_exc()}")
            return False, error_msg
            
    def execute_query(self, sql_query):
        """Execute SQL query on MySQL database"""
        try:
            conn = mysql.connector.connect(**self.db_config)
            
            # Execute query and convert to pandas DataFrame
            df = pd.read_sql_query(sql_query, conn)
            conn.close()
            
            return True, df
        except Exception as e:
            error_msg = f"Failed to execute SQL query: {str(e)}"
            self.logger.error(f"{error_msg}\n{traceback.format_exc()}")
            return False, error_msg
            
    def get_schema(self):
        """Get the database schema"""
        try:
            conn = mysql.connector.connect(**self.db_config)
            cursor = conn.cursor()
            
            # Get list of tables
            cursor.execute("SHOW TABLES;")
            tables = cursor.fetchall()
            
            schema_info = []
            
            # Get columns for each table
            for table in tables:
                table_name = table[0]
                cursor.execute(f"DESCRIBE {table_name};")
                columns = cursor.fetchall()
                
                column_info = []
                for column in columns:
                    col_name = column[0]
                    col_type = column[1]
                    column_info.append(f"{col_name} ({col_type})")
                
                schema_info.append(f"Table: {table_name}\nColumns: {', '.join(column_info)}\n")
            
            cursor.close()
            conn.close()
            
            return True, "\n".join(schema_info)
        except Exception as e:
            error_msg = f"Failed to get database schema: {str(e)}"
            self.logger.error(f"{error_msg}\n{traceback.format_exc()}")
            return False, error_msg
    
    def fix_ambiguous_columns(self, sql_query):
        """Fix ambiguous column references in the SQL query"""
        try:
            # Check if the query has JOINs (indicating potential for ambiguity)
            if " JOIN " not in sql_query.upper():
                return sql_query
            
            # Get database connection
            conn = mysql.connector.connect(**self.db_config)
            cursor = conn.cursor()
            
            # Extract table names from the query
            table_pattern = r'\bFROM\s+(\w+)|JOIN\s+(\w+)'
            tables = []
            for match in re.finditer(table_pattern, sql_query, re.IGNORECASE):
                table = match.group(1) if match.group(1) else match.group(2)
                if table:
                    tables.append(table)
            
            # Collect all columns for each table
            table_columns = {}
            for table in tables:
                try:
                    cursor.execute(f"DESCRIBE {table};")
                    columns = [row[0] for row in cursor.fetchall()]
                    table_columns[table] = columns
                except Exception as e:
                    self.logger.warning(f"Could not get columns for table {table}: {str(e)}")
                    continue
            
            # Find columns that appear in multiple tables
            all_columns = {}
            for table, columns in table_columns.items():
                for column in columns:
                    if column not in all_columns:
                        all_columns[column] = [table]
                    else:
                        all_columns[column].append(table)
            
            # Identify ambiguous columns (those that appear in multiple tables)
            ambiguous_columns = {col: tables for col, tables in all_columns.items() if len(tables) > 1}
            
            # Fix SQL query by qualifying ambiguous columns
            for col, tables in ambiguous_columns.items():
                # Choose the primary table for the column (for simplicity, use the first table)
                primary_table = tables[0]
                
                # Regular expression to find standalone column references (not already qualified)
                pattern = r'(?<!\w\.)(\b' + col + r'\b)(?!\.\w)'
                
                # Replace standalone column references with qualified references
                sql_query = re.sub(pattern, f"{primary_table}.{col}", sql_query)
            
            cursor.close()
            conn.close()
            
            return sql_query
        except Exception as e:
            # Log the error but return the original query to avoid blocking execution
            error_msg = f"Error fixing ambiguous columns: {str(e)}"
            self.logger.error(f"{error_msg}\n{traceback.format_exc()}")
            return sql_query