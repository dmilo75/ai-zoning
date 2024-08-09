import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog, scrolledtext
import os
import platform
import subprocess
import threading
import yaml

class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Python App Manager")
        self.config_path = "config.yaml"

        # Load configuration
        self.config = self.load_config(self.config_path)

        # Create Widgets
        self.create_widgets()

    def create_widgets(self):
        # Config File Section
        self.config_label = tk.Label(self.root, text="Configuration File:")
        self.config_label.pack()

        self.config_button = tk.Button(self.root, text="Load Config", command=self.load_config_file)
        self.config_button.pack()

        self.api_key_label = tk.Label(self.root, text="API Keys:")
        self.api_key_label.pack()

        self.api_key_entries = {}
        for key in ['openai_key', 'claude_key', 'gemini_key', 'census_key']:
            frame = tk.Frame(self.root)
            frame.pack()
            label = tk.Label(frame, text=f"{key}:")
            label.pack(side=tk.LEFT)
            entry = tk.Entry(frame)
            entry.insert(0, self.config.get('api_keys', {}).get(key, ''))
            entry.pack(side=tk.LEFT)
            self.api_key_entries[key] = entry

        self.save_config_button = tk.Button(self.root, text="Save Config", command=self.save_config)
        self.save_config_button.pack()

        # Conda Environment Section
        self.env_label = tk.Label(self.root, text="Conda Environment:")
        self.env_label.pack()

        self.new_env_button = tk.Button(self.root, text="Create New Environment", command=self.create_new_env)
        self.new_env_button.pack()

        self.existing_env_button = tk.Button(self.root, text="Use Existing Environment", command=self.use_existing_env)
        self.existing_env_button.pack()

        # Output Text Area
        self.output_text = scrolledtext.ScrolledText(self.root, height=20, width=80)
        self.output_text.pack()

        # Run Application Section
        self.run_button = tk.Button(self.root, text="Run Application", command=self.run_application)
        self.run_button.pack()

    def load_config_file(self):
        file_path = filedialog.askopenfilename(defaultextension=".yaml", filetypes=[("YAML files", "*.yaml"), ("All files", "*.*")])
        if file_path:
            self.config_path = file_path
            self.config = self.load_config(file_path)
            messagebox.showinfo("Info", "Configuration file loaded.")
            self.update_api_key_entries()

    def load_config(self, config_path):
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)

    def save_config(self):
        if 'api_keys' not in self.config:
            self.config['api_keys'] = {}
        for key, entry in self.api_key_entries.items():
            self.config['api_keys'][key] = entry.get()
        with open(self.config_path, 'w') as file:
            yaml.safe_dump(self.config, file)
        messagebox.showinfo("Info", "Configuration saved.")

    def create_new_env(self):
        env_name = self.get_env_name(new=True)
        if env_name:
            threading.Thread(target=self.create_and_activate_env, args=(env_name,)).start()

    def use_existing_env(self):
        env_name = self.get_env_name(new=False)
        if env_name:
            threading.Thread(target=self.activate_and_install, args=(env_name,)).start()

    def get_env_name(self, new=False):
        prompt = "Enter the name of the new conda environment:" if new else "Enter the name of the existing conda environment:"
        return simpledialog.askstring("Environment Name", prompt)

    def activate_and_install(self, env_name):
        try:
            self.log_message(f"Activating environment: {env_name}")
            self.run_command(f"conda run -n {env_name} python --version")
            self.install_dependencies(env_name)
        except subprocess.CalledProcessError:
            self.log_message(f"Failed to activate environment '{env_name}'", error=True)
            messagebox.showerror("Error", f"Failed to activate environment '{env_name}'.")

    def create_and_activate_env(self, env_name):
        try:
            self.log_message(f"Creating new environment: {env_name}")
            with open('environment.yml', 'w') as env_file:
                env_file.write(f"""
name: {env_name}
channels:
  - defaults
  - conda-forge
dependencies:
  - python=3.12.4
  - pip
  - numpy
  - pandas
  - pip:
    - -r macosrequirements.txt
""")
            self.run_command(f"conda env create -f environment.yml")
            self.log_message(f"Environment {env_name} created successfully")
            self.run_command(f"conda run -n {env_name} python --version")
        except subprocess.CalledProcessError:
            self.log_message(f"Failed to create or activate environment '{env_name}'", error=True)
            messagebox.showerror("Error", f"Failed to create or activate environment '{env_name}'.")

    def install_dependencies(self, env_name):
        machine_type = platform.system()
        requirements_file = 'macosrequirements.txt' if machine_type == 'Darwin' else 'requirements.txt'
        try:
            self.log_message(f"Installing dependencies from {requirements_file} into environment {env_name}")
            self.run_command(f"conda run -n {env_name} pip install -r {requirements_file}", timeout=1800)
            self.log_message(f"Dependencies installed successfully in environment {env_name}")
        except subprocess.TimeoutExpired:
            self.log_message(f"Timeout expired while installing dependencies in environment '{env_name}'", error=True)
            messagebox.showerror("Error", f"Timeout expired while installing dependencies in environment '{env_name}'")
        except subprocess.CalledProcessError:
            self.log_message(f"Failed to install dependencies in environment '{env_name}'", error=True)
            messagebox.showerror("Error", f"Failed to install dependencies in environment '{env_name}'.")
    # Edit this to make sure it runs with the correct command for running QA_Code_V7.py
    def run_command(self, command, timeout=None):
        try:
            result = subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=timeout)
            self.log_message(f"Running command: {command}")
            self.log_message(result.stdout)
            if result.stderr:
                self.log_message(result.stderr, error=True)
            result.check_returncode()
        except subprocess.CalledProcessError as e:
            self.log_message(e.stderr, error=True)
            raise
        except subprocess.TimeoutExpired as e:
            self.log_message("Command timed out", error=True)
            raise

    def log_message(self, message, error=False):
        self.output_text.insert(tk.END, message + "\n")
        self.output_text.see(tk.END)
        self.root.update()
        if error:
            print("ERROR: " + message)
        else:
            print(message)

    def run_application(self):
        script_path = os.path.join("code", "Main Model Code", "QA_Code_V7.py")
        script_path = f'"{script_path}"'  # Ensure the path is properly quoted
        try:
            self.log_message(f"Running the application script: {script_path}")
            self.run_command(f"python {script_path}")
            messagebox.showinfo("Info", "Application is running.")
        except subprocess.CalledProcessError:
            self.log_message(f"Failed to run the application script: {script_path}", error=True)
            messagebox.showerror("Error", f"Failed to run the application script: {script_path}")

    def update_api_key_entries(self):
        for key, entry in self.api_key_entries.items():
            entry.delete(0, tk.END)
            entry.insert(0, self.config.get('api_keys', {}).get(key, ''))

def main():
    root = tk.Tk()
    app = App(root)
    root.mainloop()

if __name__ == "__main__":
    main()
