import logging
import traceback

class SQLValidator:
    def __init__(self):
        self.logger = logging.getLogger('nl2sql')
        
        # SQL commands blacklist for security
        self.sql_blacklist = [
            "DELETE", "DROP", "UPDATE", "INSERT", "ALTER", "TRUNCATE",
            "CREATE", "RENAME", "REPLACE", "GRANT", "REVOKE"
        ]
    
    def validate(self, sql_query):
        """Validate SQL for safety"""
        try:
            sql_upper = sql_query.upper()

            # Check for blacklisted commands
            for cmd in self.sql_blacklist:
                if cmd in sql_upper and not f"'{cmd}" in sql_upper and not f'"{cmd}' in sql_upper:
                    return False, f"For security reasons, {cmd} commands are not allowed."

            # Ensure the query is a SELECT or SHOW statement
            if not (sql_upper.strip().startswith("SELECT") or sql_upper.strip().startswith("SHOW")):
                return False, "Only SELECT and SHOW queries are allowed for security reasons."

            # Ensure no multiple statements (no semicolons except at the end)
            if ";" in sql_query[:-1]:
                return False, "Multiple SQL statements are not allowed."

            return True, "SQL query is valid."
            
        except Exception as e:
            error_msg = f"Failed to validate SQL: {str(e)}"
            self.logger.error(f"{error_msg}\n{traceback.format_exc()}")
            return False, error_msg