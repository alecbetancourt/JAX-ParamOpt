# Read atom types from atomtypes.dat
with open('mobley_1019269_atomtypes.dat', 'r') as atomtypes_file:
    atomtypes = [line.strip() for line in atomtypes_file]

# Read structure.xyz and perform replacement
with open('mobley_1019269.xyz', 'r') as structure_file:
    lines = structure_file.readlines()

# Open structure.xyz for writing with modifications
with open('structure.xyz', 'w') as structure_file:
    structure_file.writelines(lines[:2])  # Write the first two lines as they are

    for i in range(2, len(lines)):
        parts = lines[i].split()
        new_line = "{} {}\n".format(atomtypes[i-2], ' '.join(parts[1:]))
        structure_file.write(new_line)
