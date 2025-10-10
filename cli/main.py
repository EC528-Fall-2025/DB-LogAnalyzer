"""
Command-line interface for FDB Log Analyzer
"""
import argparse
import sys
import os
from pathlib import Path
from service.storage import StorageService
from service.parser import LogParser


class CLI:
    """Main command-line interface class"""
    
    def __init__(self):
        self.storage_service = None
    
    def run(self, args=None):
        """Run CLI"""
        parser = self.create_parser()
        args = parser.parse_args(args)
        
        # Execute the corresponding command
        if hasattr(args, 'func'):
            args.func(args)
        else:
            parser.print_help()
    
    def create_parser(self):
        """Create command-line argument parser"""
        parser = argparse.ArgumentParser(
            prog='fdb-log-analyzer',
            description='FoundationDB Log Analysis Tool'
        )
        
        # Global arguments
        parser.add_argument(
            '--db', 
            default='fdb_logs.duckdb',
            help='Database file path (default: fdb_logs.duckdb)'
        )
        
        # Create subcommands
        subparsers = parser.add_subparsers(title='Subcommands', dest='command')
        
        # init command - Initialize database
        init_parser = subparsers.add_parser('init', help='Initialize database')
        init_parser.add_argument(
            '--schema',
            default='data/schema.sql',
            help='Database schema file path'
        )
        init_parser.set_defaults(func=self.handle_init)
        
        # load command - Load logs
        load_parser = subparsers.add_parser('load', help='Load log file(s) to database (to load mulitple, use directory)')
        load_parser.add_argument(
            'log_file',
            help='Log file path (supports JSON, XML formats)'
        )
        load_parser.add_argument(
            '--schema',
            default='data/schema.sql',
            help='Database schema file path (if database does not exist)'
        )
        load_parser.add_argument(
            '--all',
            action='store_true',
            help='If set, automatically load all rollover files (.1.xml, .2.xml, etc.) or all logs in directory'
        )
        load_parser.set_defaults(func=self.handle_load)
        
        # parse command - Parse logs (without storing)
        parse_parser = subparsers.add_parser('parse', help='Parse log file (display only, no storage)')
        parse_parser.add_argument(
            'log_file',
            help='Log file path'
        )
        parse_parser.add_argument(
            '--limit',
            type=int,
            default=10,
            help='Event display limit (default: 10)'
        )
        parse_parser.set_defaults(func=self.handle_parse)
        
        # query command - Query database
        query_parser = subparsers.add_parser('query', help='Execute SQL query')
        query_parser.add_argument(
            'sql',
            nargs='?',
            help='SQL query statement'
        )
        query_parser.add_argument(
            '--file',
            help='Read SQL query from file'
        )
        query_parser.set_defaults(func=self.handle_query)
        
        # stats command - Display statistics
        stats_parser = subparsers.add_parser('stats', help='Display database statistics')
        stats_parser.set_defaults(func=self.handle_stats)
        
        # export command - Export data
        export_parser = subparsers.add_parser('export', help='Export query results')
        export_parser.add_argument(
            '--format',
            choices=['csv', 'json', 'parquet'],
            default='csv',
            help='Export format (default: csv)'
        )
        export_parser.add_argument(
            '--output',
            required=True,
            help='Output file path'
        )
        export_parser.add_argument(
            '--query',
            default='SELECT * FROM events LIMIT 1000',
            help='SQL query for export (default: first 1000 events)'
        )
        export_parser.set_defaults(func=self.handle_export)

        # pipeline command - Run parse+storage pipeline
        pipeline_parser = subparsers.add_parser('pipeline', help='Run parse and storage pipeline')
        pipeline_parser.add_argument(
            '--input',
            required=True,
            help='Input log file or directory path'
        )
        pipeline_parser.add_argument(
            '--output',
            default='fdb_logs.duckdb',
            help='Output database or file path (default: fdb_logs.duckdb)'
        )
        pipeline_parser.add_argument(
            '--format',
            choices=['duckdb', 'csv'],
            default='duckdb',
            help='Output format (supports: duckdb, csv; default: duckdb)'
        )
        pipeline_parser.add_argument(
            '--schema',
            help='Custom database schema file path (optional)'
        )
        pipeline_parser.set_defaults(func=self.handle_pipeline)

        # rollup command - Create time-windowed aggregations
        rollup_parser = subparsers.add_parser('rollup', help='Create rollup aggregations for time windows')
        rollup_parser.add_argument(
            '--interval',
            type=int,
            default=60,
            help='Time window interval in seconds (default: 60)'
        )
        rollup_parser.set_defaults(func=self.handle_rollup)
        
        return parser
    
    def handle_init(self, args):
        """Handle init command"""
        print(f"Initializing database: {args.db}")
        
        service = StorageService(args.db)
        
        # Check if schema file exists
        if os.path.exists(args.schema):
            service.init_db(args.schema)
            print(f"Used schema file: {args.schema}")
        else:
            service.init_db()
            print("Initialized with default schema")
        
        service.close()
        print("Database initialization complete!")
    
    def handle_load(self, args):
        """Handle load command"""
        import glob, re
        # Check if log file exists
        if not os.path.exists(args.log_file):
            print(f"Error: Log file does not exist: {args.log_file}", file=sys.stderr)
            sys.exit(1)
        
        print(f"Target database: {args.db}")

        log_dir = args.log_file
        files_to_load = []

        if args.all:
            # Check if user wants to load all files
            if not os.path.isdir(log_dir):
                print(f"Error: Expected a directory, got: {log_dir}", file=sys.stderr)
                sys.exit(1)
            
            files_to_load = sorted(glob.glob((os.path.join(log_dir, "*.xml"))))
            print(f"Following {len(files_to_load)} files loaded: {files_to_load}")
            if not files_to_load:
                print(f"No .xml logs found in directory: {log_dir}")
        
        else:
            # Single file mode
            if not os.path.isfile(log_dir):
                print(f"Error: Expected a file path, got directory: {log_dir}", file=sys.stderr)
                sys.exit(1)
            files_to_load = [log_dir]


        service = StorageService(args.db)
        
        # Initialize database (if needed)
        if os.path.exists(args.schema):
            service.init_db(args.schema)
        else:
            service.init_db()
        
        # Check if data already exists
        if service.check_events_loaded():
            count = service.get_event_count()
            response = input(f"Database already contains {count} events, continue loading? (y/n): ")
            if response.lower() != 'y':
                print("Operation cancelled")
                service.close()
                return        
        
        # Load logs
        total = 0
        try:
            for file in files_to_load:
                count = service.load_logs_from_file(file, event_id_offset=total)
                total += count
            print(f"Successfully loaded {total} events from {len(files_to_load)} file(s)!")
        except Exception as e:
            print(f"Load failed: {e}", file=sys.stderr)
            sys.exit(1)
        finally:
            service.close()
    
    def handle_parse(self, args):
        """Handle parse command"""
        # Check if log file exists
        if not os.path.exists(args.log_file):
            print(f"Error: Log file does not exist: {args.log_file}", file=sys.stderr)
            sys.exit(1)
        
        print(f"Parsing log file: {args.log_file}")
        print(f"Displaying first {args.limit} events:\n")
        
        parser = LogParser()
        
        try:
            for i, event in enumerate(parser.parse_logs(args.log_file)):
                if i >= args.limit:
                    break
                
                print(f"=== Event #{event.event_id} ===")
                print(f"Time: {event.ts}")
                print(f"Severity: {event.severity}")
                print(f"Event Type: {event.event}")
                print(f"Process: {event.process}")
                print(f"Role: {event.role}")
                print(f"Machine ID: {event.machine_id}")
                print(f"Address: {event.address}")
                
                if event.fields_json:
                    print("Additional Fields:")
                    for k, v in list(event.fields_json.items())[:5]:  # Display first 5 only
                        print(f"  {k}: {v}")
                print()
        except Exception as e:
            print(f"Parse failed: {e}", file=sys.stderr)
            sys.exit(1)
    
    def handle_query(self, args):
        """Handle query command"""
        # Get SQL query
        if args.file:
            if not os.path.exists(args.file):
                print(f"Error: SQL file does not exist: {args.file}", file=sys.stderr)
                sys.exit(1)
            with open(args.file, 'r') as f:
                sql = f.read()
        elif args.sql:
            sql = args.sql
        else:
            print("Error: Please provide SQL query or use --file parameter", file=sys.stderr)
            sys.exit(1)
        
        service = StorageService(args.db)
        
        try:
            # Check if database exists
            if not os.path.exists(args.db):
                print(f"Error: Database does not exist: {args.db}", file=sys.stderr)
                print("Please use 'init' or 'load' command to create database first")
                sys.exit(1)
            
            service.init_db()
            
            # Execute query
            print(f"Executing query: {sql[:100]}{'...' if len(sql) > 100 else ''}")
            result = service.query(sql)
            
            # Display results
            df = result.df()
            print(f"\nQuery results ({len(df)} rows):")
            print(df.to_string())
            
        except Exception as e:
            print(f"Query failed: {e}", file=sys.stderr)
            sys.exit(1)
        finally:
            service.close()
    
    def handle_stats(self, args):
        """Handle stats command"""
        service = StorageService(args.db)
        
        try:
            # Check if database exists
            if not os.path.exists(args.db):
                print(f"Error: Database does not exist: {args.db}", file=sys.stderr)
                sys.exit(1)
            
            service.init_db()
            
            print(f"=== Database Statistics ===")
            print(f"Database file: {args.db}")
            
            # Get file size
            size = os.path.getsize(args.db) / (1024 * 1024)
            print(f"File size: {size:.2f} MB")
            
            # Event statistics
            total_events = service.query("SELECT COUNT(*) as count FROM events").df()['count'][0]
            print(f"\nTotal events: {total_events}")
            
            if total_events > 0:
                # Time range
                time_range = service.query("""
                    SELECT MIN(ts) as min_time, MAX(ts) as max_time 
                    FROM events 
                    WHERE ts IS NOT NULL
                """).df()
                if not time_range.empty:
                    print(f"Time range: {time_range['min_time'][0]} to {time_range['max_time'][0]}")
                
                # Severity distribution
                severity_dist = service.query("""
                    SELECT severity, COUNT(*) as count 
                    FROM events 
                    WHERE severity IS NOT NULL
                    GROUP BY severity 
                    ORDER BY severity
                """).df()
                if not severity_dist.empty:
                    print("\nSeverity distribution:")
                    for _, row in severity_dist.iterrows():
                        print(f"  Level {row['severity']}: {row['count']} events")
                
                # Top 5 event types
                top_events = service.query("""
                    SELECT event, COUNT(*) as count 
                    FROM events 
                    WHERE event IS NOT NULL
                    GROUP BY event 
                    ORDER BY count DESC 
                    LIMIT 5
                """).df()
                if not top_events.empty:
                    print("\nTop 5 event types:")
                    for _, row in top_events.iterrows():
                        print(f"  {row['event']}: {row['count']}")
                
                # Role distribution
                role_dist = service.query("""
                    SELECT role, COUNT(*) as count 
                    FROM events 
                    WHERE role IS NOT NULL
                    GROUP BY role 
                    ORDER BY count DESC
                    LIMIT 10
                """).df()
                if not role_dist.empty:
                    print("\nRole distribution (Top 10):")
                    for _, row in role_dist.iterrows():
                        print(f"  {row['role']}: {row['count']} events")
            
        except Exception as e:
            print(f"Failed to get statistics: {e}", file=sys.stderr)
            sys.exit(1)
        finally:
            service.close()
    
    def handle_export(self, args):
        """Handle export command"""
        service = StorageService(args.db)
        
        try:
            # Check if database exists
            if not os.path.exists(args.db):
                print(f"Error: Database does not exist: {args.db}", file=sys.stderr)
                sys.exit(1)
            
            service.init_db()
            
            print(f"Executing query: {args.query[:100]}{'...' if len(args.query) > 100 else ''}")
            result = service.query(args.query)
            df = result.df()
            
            print(f"Exporting {len(df)} rows to {args.output}")
            
            # Export based on format
            if args.format == 'csv':
                df.to_csv(args.output, index=False)
            elif args.format == 'json':
                df.to_json(args.output, orient='records', indent=2)
            elif args.format == 'parquet':
                df.to_parquet(args.output, index=False)
            
            print(f"Successfully exported to: {args.output}")
            
        except Exception as e:
            print(f"Export failed: {e}", file=sys.stderr)
            sys.exit(1)
        finally:
            service.close()

    def handle_pipeline(self, args):
        """Handle pipeline command"""
        input_path = Path(args.input).expanduser()
        output_path = Path(args.output).expanduser()
        format_value = args.format.lower()

        if not input_path.exists():
            print(f"Error: Input path does not exist: {input_path}", file=sys.stderr)
            sys.exit(1)

        if format_value != 'duckdb':
            print("Error: Currently only DuckDB output format is supported", file=sys.stderr)
            sys.exit(1)

        schema_path = None
        if args.schema:
            schema_candidate = Path(args.schema).expanduser()
            if not schema_candidate.exists():
                print(f"Warning: Specified schema file not found {schema_candidate}, will use default schema")
            else:
                schema_path = schema_candidate
        else:
            default_schema = Path('data/schema.sql')
            if default_schema.exists():
                schema_path = default_schema

        target_files = []
        if input_path.is_dir():
            for file_path in sorted(input_path.rglob('*')):
                if file_path.is_file() and self._is_supported_log_file(file_path):
                    target_files.append(file_path)
            if not target_files:
                print(f"Error: No supported log files found in directory {input_path}", file=sys.stderr)
                sys.exit(1)
        elif input_path.is_file():
            if self._is_supported_log_file(input_path):
                target_files.append(input_path)
            else:
                print(f"Error: Unsupported log file type: {input_path.suffix}", file=sys.stderr)
                sys.exit(1)
        else:
            print(f"Error: Invalid input path: {input_path}", file=sys.stderr)
            sys.exit(1)

        service = StorageService(str(output_path))

        try:
            if schema_path is not None:
                service.init_db(str(schema_path))
                print(f"Used schema file: {schema_path}")
            else:
                service.init_db()
                print("No schema file specified, using default schema")

            try:
                existing_count = service.get_event_count()
            except Exception:
                existing_count = 0

            total_loaded = 0
            for idx, file_path in enumerate(target_files, start=1):
                print(f"[{idx}/{len(target_files)}] Processing: {file_path}")
                offset = existing_count + total_loaded
                try:
                    loaded = service.load_logs_from_file(str(file_path), event_id_offset=offset)
                except Exception as exc:
                    print(f"Failed to process file: {file_path} -> {exc}", file=sys.stderr)
                    raise
                total_loaded += loaded
                print(f"  Loaded {loaded} events (cumulative: {total_loaded})")

            print("\n=== Pipeline Execution Complete ===")
            print(f"Input files: {len(target_files)}")
            print(f"New events: {total_loaded}")
            print(f"Output database: {output_path}")

        finally:
            service.close()

    def handle_rollup(self, args):
        """Handle rollup command"""
        service = StorageService(args.db)
        
        try:
            # Check if database exists
            if not os.path.exists(args.db):
                print(f"Error: Database does not exist: {args.db}", file=sys.stderr)
                print("Please load data first using 'load' or 'pipeline' command")
                sys.exit(1)
            
            service.init_db()
            
            # Check if there are events
            event_count = service.get_event_count()
            if event_count == 0:
                print("Error: No events in database", file=sys.stderr)
                print("Please load data first using 'load' or 'pipeline' command")
                sys.exit(1)
            
            print(f"Creating rollup aggregations with {args.interval}s time windows...")
            print(f"Processing {event_count} events...")
            
            # Call Group B's rollup function
            service.create_rollups(args.interval)
            
            print(f"\nRollup table 'rollups_{args.interval}s' created successfully!")
            print(f"\nYou can now query the rollup table:")
            print(f"  python main.py query \"SELECT * FROM rollups_{args.interval}s LIMIT 10\"")
            
        except Exception as e:
            print(f"Rollup creation failed: {e}", file=sys.stderr)
            sys.exit(1)
        finally:
            service.close()

    @staticmethod
    def _is_supported_log_file(file_path: Path) -> bool:
        """Check if log file is a supported type"""
        supported_suffixes = {'.json', '.xml', '.log', '.txt'}
        return file_path.suffix.lower() in supported_suffixes


def cli():
    """CLI entry function"""
    cli_instance = CLI()
    cli_instance.run()
