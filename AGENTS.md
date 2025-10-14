# FDB Log Analyzer - Agent Guidelines

## Build/Lint/Test Commands

### Testing
- Run all tests: `python test_cli.py`
- Run single test: `python -m pytest test_cli.py::test_function_name -v`
- Test specific CLI functionality: `python test_cli.py`

### Linting & Code Quality
- No linting tools configured - use Python's built-in tools
- Check syntax: `python -m py_compile file.py`
- Basic style check: `python -c "import ast; ast.parse(open('file.py').read())"`

### Running the Application
- Main CLI: `python main.py --help`
- Initialize database: `python main.py init --schema data/schema.sql`
- Parse logs: `python main.py parse data/sample_log.json --limit 5`
- Run pipeline: `python main.py pipeline --input data/sample_log.json --output test.duckdb`

## Code Style Guidelines

### Imports
- Standard library imports first
- Third-party imports second
- Local imports last
- Use absolute imports
- Group imports with blank lines between groups

```python
import os
import sys
from pathlib import Path

import duckdb
import pandas as pd
from pydantic import BaseModel

from dto.event import EventModel
from service.parser import LogParser
```

### Naming Conventions
- Functions and variables: `snake_case`
- Classes: `PascalCase`
- Constants: `UPPER_SNAKE_CASE`
- Private methods: `_leading_underscore`

### Type Hints
- Use typing module for all function parameters and return values
- Use Optional for nullable types
- Use Union for multiple possible types

```python
from typing import Optional, Dict, Any, Generator

def parse_logs(self, path: str) -> Generator[EventModel, None, None]:
    # implementation

def get_event(self, event_id: str) -> Optional[EventModel]:
    # implementation
```

### Error Handling
- Use try/except blocks for expected errors
- Log errors with descriptive messages
- Use sys.exit(1) for CLI command failures
- Raise custom exceptions for business logic errors

```python
try:
    result = service.load_logs_from_file(args.log_file)
except Exception as e:
    print(f"Load failed: {e}", file=sys.stderr)
    sys.exit(1)
```

### Documentation
- Use docstrings for all classes and public methods
- Follow Google/NumPy docstring format
- Document parameters, return values, and exceptions

```python
class LogParser:
    """Log parsing service for FDB trace logs.

    Supports both JSON and XML formats with automatic format detection.
    """

    def parse_logs(self, path: str) -> Generator[EventModel, None, None]:
        """Parse log file and yield EventModel instances.

        Args:
            path: Path to the log file (JSON or XML format)

        Yields:
            EventModel instances for each parsed log event

        Raises:
            FileNotFoundError: If the log file doesn't exist
            ValueError: If the log format is unsupported
        """
```

### Code Structure
- Use classes for services and complex logic
- Keep functions focused on single responsibilities
- Use dataclasses/Pydantic models for data structures
- Separate CLI logic from business logic

### Database Operations
- Use context managers for database connections when possible
- Close connections explicitly in finally blocks
- Use parameterized queries to prevent SQL injection
- Handle database errors gracefully

```python
service = StorageService(args.db)
try:
    service.init_db()
    result = service.query(sql)
finally:
    service.close()
```

### File Operations
- Use pathlib.Path for file path operations
- Check file existence before operations
- Use with statements for file handles
- Handle encoding explicitly for text files

### CLI Design
- Use argparse for command-line interfaces
- Provide helpful error messages
- Use consistent exit codes (0 for success, 1 for errors)
- Support --help for all commands