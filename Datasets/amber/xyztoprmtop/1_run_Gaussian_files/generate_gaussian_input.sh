#!/bin/bash

# Function to generate Gaussian input file with .com extension
generate_gaussian_input() {
    local xyz_file="$1"
    local com_file="${xyz_file%.xyz}.com"

    # Check if the input XYZ file exists
    if [ ! -f "$xyz_file" ]; then
        echo "Error: Input XYZ file not found!"
        exit 1
    fi

    # Check if the XYZ file contains atomic coordinates
    if [ $(grep -cE '^[A-Za-z]+\s+(-?[0-9]+(\.[0-9]+)?\s+){2}-?[0-9]+(\.[0-9]+)?$' "$xyz_file") -eq 0 ]; then
        echo "Error: Invalid XYZ file format. Please provide XYZ coordinates."
        exit 1
    fi

    # Create the Gaussian input file with .com extension
    cat <<EOF >"$com_file"
%chk=${com_file%.com}.chk
%mem=1GB
%NProcShared=1

#HF/6-31G* SCF=tight Pop=MK iop(6/33=2) iop(6/42=6) opt

${com_file%.com}

0 1
EOF

    # Append the XYZ coordinates to the Gaussian input file
    cat "$xyz_file" >>"$com_file"

    echo "Gaussian input file $com_file generated successfully."
}

# Call the function and pass the XYZ file as an argument
generate_gaussian_input "FFFF.xyz"
