#!/bin/bash

# Upgrade pip to the latest version
echo "Upgrading pip..."
pip install --upgrade pip

# Check if packages.txt exists
if [[ ! -f packages.txt ]]; then
    echo "Error: packages.txt file not found!"
    exit 1
fi

# Install all packages listed in packages.txt
echo "Installing packages from packages.txt..."
while IFS= read -r package; do
    if [[ -n "$package" ]]; then
        echo "Installing $package..."
        pip install "$package"
    fi
done < packages.txt

echo "All packages installed."
