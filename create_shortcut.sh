#!/bin/bash

# Get the current directory
CURRENT_DIR=$(pwd)

# Create the shortcut script
cat << EOF > ~/Desktop/octopi_malaria.sh
#!/bin/bash

# Change to the script directory
cd $CURRENT_DIR

# Run the Python script
python3 run.py
EOF

# Create another shortcut for the simulation
cat << EOF > ~/Desktop/octopi_malaria_simulation.sh
#!/bin/bash

# Change to the script directory
cd $CURRENT_DIR

# Run the Python script
python3 run.py simulation
EOF

# Make the shortcut script executable
chmod +x ~/Desktop/octopi_malaria.sh
chmod +x ~/Desktop/octopi_malaria_simulation.sh

echo "Desktop shortcut created: ~/Desktop/octopi_malaria.sh"
echo "You can now right click this file, and run as program on your desktop to run the program."