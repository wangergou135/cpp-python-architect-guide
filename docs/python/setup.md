# Python Environment Setup Guide

## Introduction
This guide will help you set up a Python development environment on your machine. 

## Prerequisites
- Ensure that you have administrative access to your machine.
- For Windows users, it is recommended to use Windows Subsystem for Linux (WSL).

## Step 1: Install Python
1. Download the latest version of Python from the [official Python website](https://www.python.org/downloads/).
2. Follow the installation instructions for your operating system.
3. Make sure to check the box that says "Add Python to PATH" during installation.

## Step 2: Verify Python Installation
Open your terminal or command prompt and run the following command:
```bash
python --version
```
This should display the installed version of Python.

## Step 3: Install Virtual Environment
It's a good practice to use a virtual environment for your Python projects. To install `virtualenv`, run:
```bash
pip install virtualenv
```

## Step 4: Create a Virtual Environment
Navigate to your project directory and create a new virtual environment:
```bash
mkdir my_project
cd my_project
virtualenv venv
```

## Step 5: Activate the Virtual Environment
- On Windows:
```bash
venv\Scripts\activate
```
- On macOS/Linux:
```bash
source venv/bin/activate
```

## Step 6: Install Required Packages
You can now install any required packages using pip. For example:
```bash
pip install requests
```

## Conclusion
You have successfully set up your Python environment. 
Timestamp: 2025-08-10 10:37:47 UTC
