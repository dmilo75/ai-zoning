# README

## Summary
This README provides instructions for installing Miniconda on Windows and macOS/Unix systems. It also includes steps to create and activate a new conda environment with Python 3.12.4, as well as instructions to do the same using venv. Finally, it covers how to install required packages from `macosrequirements.txt` and `requirements.txt`.

## Installing Miniconda

### Windows
1. Download the Miniconda installer for Windows from the [official website](https://docs.conda.io/en/latest/miniconda.html).
2. Run the installer and follow the prompts to complete the installation.
3. During installation, you can choose to add Miniconda to your PATH environment variable.

### macOS/Unix
1. Download the Miniconda installer for macOS/Unix from the [official website](https://docs.conda.io/en/latest/miniconda.html).
2. Open a terminal and navigate to the directory where the installer was downloaded.
3. Run the following command to install Miniconda:

```bash
bash Miniconda3-latest-MacOSX-x86_64.sh
```

4. Follow the prompts to complete the installation.

## Creating and Activating a New Conda Environment

<details>
<summary>Conda Environment</summary>

### Steps
1. Open a terminal (macOS/Unix) or Anaconda Prompt (Windows).
2. Create a new conda environment with Python 3.12.4:
   
   ```bash
   conda create -n myenv python=3.12.4
   ```

3. Activate the environment:
   
   ```bash
   conda activate myenv
   ```

4. Install the required packages:
   
   For macOS:
   ```bash
   pip install -r macosrequirements.txt
   ```

   For Windows:
   ```bash
   pip install -r requirements.txt
   ```

</details>

<details>
<summary>venv Environment</summary>

### Steps
1. Open a terminal (macOS/Unix) or Command Prompt (Windows).
2. Create a new virtual environment with venv:
   
   ```bash
   python -m venv myenv
   ```

3. Activate the environment:
   
   On macOS/Unix:
   ```bash
   source myenv/bin/activate
   ```

   On Windows:
   ```bash
   myenv\Scripts\activate
   ```

4. Install the required packages:
   
   For macOS:
   ```bash
   pip install -r macosrequirements.txt
   ```

   For Windows:
   ```bash
   pip install -r requirements.txt
   ```

</details>