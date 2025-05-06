# NL2SQL: Natural Language to SQL Query System

A modular application that converts natural language questions into SQL queries using OpenAI's language models.

## Table of Contents

- [Overview](#overview)
- [System Architecture](#system-architecture)
- [Module Details](#module-details)
  - [Main Module](#main-module)
  - [Database Module](#database-module)
  - [NLP Module](#nlp-module)
  - [UI Module](#ui-module)
  - [Utils Module](#utils-module)
  - [Visualization Module](#visualization-module)
- [Configuration](#configuration)
- [Security Features](#security-features)
- [Database Schema Requirements](#database-schema-requirements)
- [Development and Extension](#development-and-extension)
- [Troubleshooting](#troubleshooting)
- [Technical Requirements](#technical-requirements)
- [Web Frontend](#web-frontend)

## Overview

NL2SQL is a modular Python application that allows users to query MySQL databases using natural language. The application leverages OpenAI's language models to translate user questions into valid SQL queries, executes them on the connected database, and presents the results with visualizations and AI-generated insights.

## System Architecture

The application follows a modular architecture with clear separation of concerns:

```
nl2sql/
├── __init__.py           # Package initialization
├── main.py               # Main entry point
├── database/             # Database connection and query execution
│   ├── __init__.py
│   └── connector.py      # Database operations
├── nlp/                  # Natural language processing
│   ├── __init__.py
│   └── sql_generator.py  # AI-based SQL generation
├── ui/                   # User interface components
│   ├── __init__.py
│   ├── main_window.py    # Primary application UI
│   ├── results_view.py   # Results display widgets
│   └── settings_dialog.py# Configuration UI
├── utils/                # Utility modules
│   ├── __init__.py
│   ├── config.py         # Configuration management
│   ├── logger.py         # Logging functionality
│   └── sql_validator.py  # SQL security validation
├── visualization/        # Data visualization
│   ├── __init__.py
│   └── chart_generator.py # Chart creation from query results
└── web_frontend/         # PHP-based web frontend
    ├── index.php         # Main entry point for the web application
    └── js/app.js         # JavaScript for client-side interactivity
```

## Module Details

### Main Module

**File: `main.py`**

The main entry point that initializes and coordinates all application components.

**Key Functions:**
- `main()`: Initializes the application, sets up logging, loads configuration, creates component instances, and starts the UI.

**Technical Implementation:**
- Creates component instances in a specific order to ensure proper dependency injection
- Implements global exception handling to prevent crashes
- Manages application lifecycle through the Tkinter event loop

### Database Module

**File: `database/connector.py`**

Handles all database operations including connections, query execution, and schema retrieval.

**Key Functions:**
- `test_connection()`: Tests if the database connection is valid
- `execute_query()`: Executes SQL queries and returns results as a pandas DataFrame
- `get_schema()`: Retrieves the database schema for context in query generation
- `fix_ambiguous_columns()`: Resolves ambiguous column references in SQL queries

**Technical Implementation:**
- Uses `mysql-connector-python` for database communication
- Converts query results to pandas DataFrames for easier data manipulation
- Implements error handling with detailed logging
- Uses regular expressions to detect and fix ambiguous column references in JOINs

### NLP Module

**File: `nlp/sql_generator.py`**

Connects to OpenAI's language models to convert natural language to SQL and generate insights.

**Key Functions:**
- `generate_sql()`: Converts natural language questions to SQL queries using AI
- `generate_summary()`: Creates insightful summaries of query results

**Technical Implementation:**
- Uses the OpenAI API client to communicate with language models
- Constructs specialized prompts that include database schema information
- Implements temperature control for deterministic SQL generation
- Handles API errors and rate limiting
- Enforces best practices for SQL generation, such as column disambiguation

### UI Module

**Files:**
- `ui/main_window.py`: Main application window
- `ui/results_view.py`: Handles displaying query results
- `ui/settings_dialog.py`: Configuration interface

**Key Functions:**
- `MainWindow`: Creates and manages the main application interface
- `ResultsView`: Displays query results in table, chart, and summary formats
- `SettingsDialog`: Provides interface for database and API configuration

**Technical Implementation:**
- Built with Tkinter and ttk for platform-compatible UI
- Uses the Notebook widget to organize results views
- Implements threaded query execution to maintain UI responsiveness
- Includes example queries for user reference
- Uses scrollable text and tree views for data display

### Utils Module

**Files:**
- `utils/config.py`: Configuration management
- `utils/logger.py`: Logging setup and management
- `utils/sql_validator.py`: SQL security validation

**Key Functions:**
- `Config`: Loads, stores, and saves application configuration
- `setup_logger()`: Configures application logging
- `SQLValidator.validate()`: Validates SQL queries for security risks

**Technical Implementation:**
- Uses JSON for configuration persistence
- Implements logging with level control and file output
- Blacklists dangerous SQL commands
- Validates queries to ensure read-only operations

### Visualization Module

**File: `visualization/chart_generator.py`**

Generates visualizations from query results.

**Key Functions:**
- `generate_chart()`: Creates appropriate charts based on query results

**Technical Implementation:**
- Uses Matplotlib for chart generation
- Determines the appropriate chart type based on data characteristics
- Embeds charts in Tkinter using the FigureCanvasTkAgg interface
- Handles edge cases like empty datasets or non-numeric data

## Configuration

The application uses a JSON configuration file (`nl2sql_config.json`) with the following structure:

```json
{
  "database": {
    "host": "localhost",
    "user": "nl2sql_user",
    "password": "password",
    "database": "nl2sql_test",
    "port": 3306
  },
  "openai_api_key": "your-api-key",
  "ai_model": "gpt-4o-mini"
}
```

## Security Features

The application includes several security measures:

1. **SQL Validation**: The `SQLValidator` class prevents execution of dangerous commands
2. **Read-Only Operations**: Only SELECT and SHOW statements are allowed
3. **Blacklist Protection**: Commands like DELETE, DROP, UPDATE, INSERT, etc. are blocked
4. **Multi-Statement Prevention**: Only single SQL statements are permitted
5. **Input Validation**: User inputs are validated before processing

## Database Schema Requirements

For optimal SQL generation, the database should:

1. Have well-defined primary and foreign keys
2. Use consistent naming conventions
3. Include proper data types for columns
4. Have tables with descriptive names

## Development and Extension

This modular architecture makes extending the application straightforward:

1. **Adding new AI models**: Extend the `SQLGenerator` class to support additional models
2. **Supporting more databases**: Create new connector classes in the database module
3. **Enhanced visualizations**: Extend the `ChartGenerator` class with more chart types
4. **Custom UI themes**: Modify the UI classes with custom styling

## Troubleshooting

Common issues and solutions:

1. **Connection Issues**: Verify MySQL server is running and credentials are correct
2. **API Key Errors**: Check that your OpenAI API key is valid and has sufficient credits
3. **SQL Generation Problems**: Try rephrasing your question or check the database schema
4. **No Data Returned**: Ensure your query makes sense for the database structure
5. **Visualization Issues**: Some data structures may not generate ideal charts automatically

## Technical Requirements

- Python 3.7+
- MySQL/MariaDB database
- OpenAI API key (for GPT-4o-mini access)
- Required Python packages:
  - mysql-connector-python
  - pandas
  - matplotlib
  - python-dotenv
  - openai
  - tkinter (included with standard Python)

## Web Frontend

The `web_frontend` directory contains the PHP-based frontend for the NL2SQL application. This frontend allows users to interact with the application through a web browser.

### Files and Structure

- `index.php`: The main entry point for the web application.
- `js/app.js`: Contains JavaScript code for client-side interactivity.

### Hosting the Web Frontend

To host the web frontend using XAMPP:

1. **Install XAMPP**:
   - Download XAMPP from [Apache Friends](https://www.apachefriends.org/).
   - Follow the installation instructions for your operating system.

2. **Start XAMPP Services**:
   - Start the Apache and MySQL services using the XAMPP control panel or the command line:
     ```bash
     sudo /opt/lampp/lampp start
     ```

3. **Deploy the Web Frontend**:
   - Copy the contents of the `web_frontend` directory to the `htdocs` folder of your XAMPP installation:
     ```bash
     sudo cp -r /path/to/web_frontend /opt/lampp/htdocs/nl2sql
     ```

4. **Set Permissions**:
   - Ensure the files have the correct permissions:
     ```bash
     sudo chmod -R 755 /opt/lampp/htdocs/nl2sql
     ```

5. **Access the Application**:
   - Open your browser and navigate to `http://localhost/nl2sql/`.

### Integration with Backend

The web frontend communicates with the backend Python application to process natural language queries and return SQL results. Ensure the backend is running and accessible for the frontend to function correctly.

### Customization

- Modify `index.php` to change the layout or functionality of the web interface.
- Update `js/app.js` to add or enhance client-side interactivity.