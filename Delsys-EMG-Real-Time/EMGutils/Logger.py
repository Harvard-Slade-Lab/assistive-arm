import json
## so far not used!
class LogManager:
    def __init__(self, log_filename):
        self.log_filename = log_filename
        self.log_entries = []
    
    def append_log_entry(self, log_entry):
        # Append the new log entry to the list
        self.log_entries.append(log_entry)

        # Save the log_entries to the file after every append (or periodically)
        self.save_logs()
    
    def save_logs(self):
        # Save the log entries to the JSON file
        try:
            with open(self.log_filename, 'w') as f:
                json.dump(self.log_entries, f, indent=4)
        except Exception as e:
            print(f"Error saving logs: {e}")

# Example usage
log_manager = LogManager('log_entries.json')

# Sample log entry
log_entry = {
    'Tag': 'Assistive Profile',
    'Segment Start Index': 0,
    'Segment End Index': 100,
    'Start Time': 12345,
    'End Time': 12355,
    'Score': 0
}

# Append log entry and save it immediately
log_manager.append_log_entry(log_entry)

# More log entries can be added similarly
