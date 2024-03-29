# Convert a geo file into a series of amber format prmtop files
def generate_prmtop(geo, out_dir):
    # Split geo into individual structures to convert
    geometries = [[]]
    idx = 0
    with open(geo, 'r') as f:
        for line in f:
            if len(line.strip()) == 0 :
                idx += 1
                geometries.append([])
            
            geometries[idx].append(line)
    
    for idx, geometry in enumerate(geometries):
        with open(out_dir+'/'+str(idx)+'.pdb', "w") as f:
            for line in geometry:
                # Convert to remark
                if 'DESCRP' in line[:6]:
                    f.write("REMARK " + line[6:])
                # bgf format : 'ATOM'|'HETATM',1X,I5,1X,A5,1X,A3,1X,A1,1X,A5,3F10.5,1X,A5,I3,I2,1X,F8.5
                # pdb format : https://www.cgl.ucsf.edu/chimera/docs/UsersGuide/tutorials/pdbintro.html
                # atom #, atom label, residue name, chain designator, residue #, x, y, z (in A),
                # atom type, max # cov bonds, # lone pairs, atomic charge
                elif 'HETATM' in line[:6]:
                    atm_num = line[7:12].strip()
                    #label should technically center right with 3 characters instead of left
                    atm_lbl = line[13:18].strip().upper()
                    res_name = line[19:22].strip().upper()
                    chn_des = line[24]
                    res_num = line[25:30].strip()
                    x = float(line[30:40].strip())
                    y = float(line[40:50].strip())
                    z = float(line[50:60].strip())
                    atm_tpe = line[61:66].strip()
                    cov_bnd = line[66:69].strip()
                    lne_prs = line[69:71].strip()
                    atm_chg = line[72:80].strip()
                    f.write(f'HETATM{atm_num:>5}{"":1}{atm_lbl:^4}{"":1}{res_name:>3}{"":1}{chn_des:1}{res_num:>4}{"":4}{x:>8.3f}{y:>8.3f}{z:>8.3f}{1.0:>6.2f}{0.0:>6.2f}')
                    f.write('\n')
                elif 'END' in line[:6]:
                    f.write("END")
    return

# Take geo file as input and generate list of freesolv molecules
# Creates tleap input script to generate prmtops from freesolv gaff mol2 files
def match_geo_to_freesolv(geo, out_dir, freesolv_dir):
    return

# Build list of prmtop files from DESCRP section in geo file
# This should eventually be automated with bgf or mol2 files
def build_prm_list(geo, prm_dir):
    flist = []
    with open(geo, 'r') as f:
        for line in f:
            if 'DESCRP' in line[:6]:
                flist.append(prm_dir + '/' + line[6:].strip() + '.prmtop')

    return flist

def main(geo, out_dir):
    generate_prmtop(geo, out_dir)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Create an AMBER prmtop file from a geo file')
    parser.add_argument('--geo', metavar='path',
                        required=True,
                        help='path to the file with the list of geometries')
    parser.add_argument('--out_dir', metavar='path',
                        required=True,
                        help='path to output directory')
    args = parser.parse_args()
    main(args.geo, args.out_dir)