#!/bin/bash

# Get the current directory
CURRENT_DIR=$(pwd)

# Create the shortcut script
cat << EOF > $CURRENT_DIR/go.sh
#!/bin/bash
export BUCKET_NAME="\$(grep BUCKET_NAME ~/.bashrc | cut -d '=' -f2 | tr -d '"')"
export SERVICE_ACCOUNT_JSON_KEY="\$(grep SERVICE_ACCOUNT_JSON_KEY ~/.bashrc | cut -d '=' -f2 | tr -d '"')"
# Change to the script directory
cd $CURRENT_DIR

# Run the Python script
python3 run.py
EOF

# Create another shortcut for the simulation
cat << EOF > ~/Desktop/octopi_malaria.desktop
[Desktop Entry]
Type=Application
Name=Octopi_malaria
Comment=Execute the Octopi Malaria analysis script
Exec=gnome-terminal -- bash -c "$CURRENT_DIR/go.sh; exec bash"
Icon=$CURRENT_DIR/checkpoint/cephla_logo.svg
Terminal=true
Categories=Science;
EOF

# Make the shortcut script executable
chmod +x $CURRENT_DIR/go.sh
chmod +x ~/Desktop/octopi_malaria.desktop

echo "Desktop shortcut created: ~/Desktop/octopi_malaria.desktop"
echo "You can now right click this file, and run as program on your desktop to run the program."