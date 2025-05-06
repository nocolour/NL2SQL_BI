import openai
import logging
import traceback

class SQLGenerator:
    def __init__(self, api_key, model_name):
        self.api_key = api_key
        self.model_name = model_name
        self.logger = logging.getLogger('nl2sql')
        
    def generate_sql(self, natural_query, schema_info):
        """Generate SQL from natural language using selected AI model"""
        try:
            # Set up the prompt for AI model with improved instructions for column disambiguation
            prompt = f"""
            You are a natural language to SQL converter. Convert the following question into a SQL query for MySQL.
            
            Database schema:
            {schema_info}
            
            Question: {natural_query}
            
            Important guidelines:
            1. Only return the SQL query without any explanation or markdown formatting
            2. Do not use backticks or any other formatting
            3. Only use SELECT statements or SHOW statements for security
            4. Your response should be a valid SQL query that can be executed directly
            5. Keep it simple and focused on answering the question
            6. ALWAYS use fully qualified column names (table_name.column_name) in SELECT, JOIN, WHERE, GROUP BY, 
               and ORDER BY clauses when the query involves multiple tables
            7. Be particularly careful with JOIN operations to avoid ambiguous column references
            
            SQL Query:
            """

            client = openai.OpenAI(api_key=self.api_key)
            response = client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a natural language to SQL converter. You output only valid SQL queries with fully qualified column names."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=200
            )

            # Extract SQL query from response
            sql_query = response.choices[0].message.content.strip()

            # Add semicolon if missing
            if not sql_query.endswith(';'):
                sql_query += ';'

            return True, sql_query
        except Exception as e:
            error_msg = f"Failed to generate SQL query: {str(e)}"
            self.logger.error(f"{error_msg}\n{traceback.format_exc()}")
            return False, error_msg
    
    def generate_summary(self, natural_query, sql_query, df):
        """Generate summary of the query results using selected AI model"""
        try:
            # If we don't have any data, return a simple message
            if df.empty:
                return True, "No data found for your query."

            # Get data statistics
            row_count = len(df)
            
            # Create a summary of the data
            data_sample = df.head(5).to_string()
            data_stats = df.describe().to_string() if not df.empty else "No data"

            # Set up the prompt for AI model
            prompt = f"""
            Analyze the following database query and results:
            
            Natural Language Query: {natural_query}
            SQL Query: {sql_query}
            
            Data sample (first 5 rows):
            {data_sample}
            
            Data statistics:
            {data_stats}
            
            Total rows returned: {row_count}
            
            Please provide a concise, meaningful summary of these results in 3-4 sentences. 
            Focus on key insights, patterns, or notable findings in the data.
            """

            # Call AI model for summary
            client = openai.OpenAI(api_key=self.api_key)
            response = client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You provide concise, insightful summaries of database query results."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.5,
                max_tokens=200
            )

            # Extract summary from response
            summary = response.choices[0].message.content.strip()

            return True, summary
        except Exception as e:
            error_msg = f"Failed to generate summary: {str(e)}"
            self.logger.error(f"{error_msg}\n{traceback.format_exc()}")
            return False, error_msg