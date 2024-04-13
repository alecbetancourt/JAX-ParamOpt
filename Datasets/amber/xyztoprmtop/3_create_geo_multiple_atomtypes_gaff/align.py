def align_columns(input_file, output_file):
    with open(input_file, 'r') as f:
        lines = f.readlines()

    # Find the maximum width for each column
    widths = [0] * len(lines[2].split())
    for line in lines[2:]:
        columns = line.split()
        for i, column in enumerate(columns):
            widths[i] = max(widths[i], len(column))

    # Align the columns and write to the output file
    with open(output_file, 'w') as f:
        for line in lines:
            if line.startswith('#'):
                # Write comment lines as is
                f.write(line)
            elif line.strip():
                columns = line.split()
                aligned_line = ' '.join(column.ljust(width) for column, width in zip(columns, widths))
                f.write(aligned_line + '\n')

# Usage example
input_file = 'input.xyz'
output_file = 'output.xyz'
align_columns(input_file, output_file)

