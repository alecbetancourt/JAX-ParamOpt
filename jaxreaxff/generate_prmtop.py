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

def read_parameter_file(params_file, ignore_sensitivity=1):
    if not os.path.exists(params_file):
        return
    params = []
    f = open(params_file, 'r')

    for line in f:
        param = []
        line = line.split('!')[0]
        split_line = line.strip().split()
        if split_line[1] == 'torsion':
            param.append(int(split_line[0])) #geo_idx
            #might need to eventually map this to int for jax compatibility
            #e.g. charge = 1, torsion = 2, etc
            param.append(1) #prm_name torsion = 1
            param.append(float(split_line[2])) #sensitivity
            #might also need to create a mapping scheme between atom names and indicies
            param.append(int(split_line[3])) #t1
            param.append(int(split_line[4])) #t2
            param.append(int(split_line[5])) #t3
            param.append(int(split_line[6])) #t4
            param.append(float(split_line[7])) #k_low
            param.append(float(split_line[8])) #k_high
            param.append(float(split_line[9])) #period_low
            param.append(float(split_line[10])) #period_high
            param.append(float(split_line[11])) #phase_low
            param.append(float(split_line[12])) #phase_high
            param.append(float(split_line[13])) #scee_low
            param.append(float(split_line[14])) #scee_high
            param.append(float(split_line[15])) #scnb_low
            param.append(float(split_line[16])) #scnb_high
            params.append(param)
    
    return params


#function that takes aligned forcefields and uses parmed to save final parameters to modified
def parse_and_save_force_field_amber(final_params, param_list, ff_sizes_dict, prm_flist):
    #final params will be 

    #size dict should be list of dictionaries
    #e.g. for 1...N : sizedicts[i]["b_k"] = size(b_k)
    
    # idx to keep track of location in final parameter list
    i = 0

    for i, prm in enumerate(params_list):
        if prm[1] == "bond":
            return
        if prm[1] == 1: # torsion = 1
            # torsion format
            # geo_idx "torsion" sensitivity t1 t2 t3 t4 klow high perlow high phaselow high sceelow high scnblow high
            parmed_buffer='parm %s\n' % (prm_flist[prm[0]])
            #prmtop = AmberPrmtopFile()
            atom_mask = [prmtop["atom_names"][a] for a in prm[5:8]]
            #atom_mask = prm[5:8]
            params = final_params[i:i+5]
            #params=[float(k), float(per), float(phase),  float(scee), float(scnb)]
            parmed_buffer += 'deleteDihedral @%s @%s @%s @%s' % tuple(atom_mask) + '\n'
            parmed_buffer += 'addDihedral @%s @%s @%s @%s %f %f %f %f %f' % (tuple(atom_mask+params)) + '\n'
            parmed_buffer+='outparm %s.new\n' % (prmtop_file_name)
            command=['parmed']
            result = subprocess.run(command, input=parmed_buffer, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            i += 5
            #if debug then print these reults at some point

    #copy all prmtops to ff_out
    return

def get_params_amber(aligned_ff, params_list, size_dicts):
    #aligned ff is # clusters long
    params = []
    for i, prm in enumerate(params_list):
        if prm[1] == 1: # torsion = 1
            #add dihedral parameter index to params list to avoid having to find it every time
            #either have it in the params file or find it with argwhere
            prm_idx = prm[1] # this idx may change
            geo_idx = prm[0]
            #TODO: add dynamic cluster sizes here
            cluster_num = geo_idx % 40
            k = aligned_ff[cluster_num]["t_k"][geo_idx][prm_idx]
            period = aligned_ff[cluster_num]["t_period"][geo_idx][prm_idx]
            phase = aligned_ff[cluster_num]["t_phase"][geo_idx][prm_idx]
            scee = aligned_ff[cluster_num]["scee"][geo_idx][prm_idx]
            scnb = aligned_ff[cluster_num]["scnb"][geo_idx][prm_idx]
            params.extend([k, period, phase, scee, scnb])

    return params

def set_params_amber(aligned_ff, params_list, size_dicts, params):
    #grab params keeping counter
    #e.g. if param_list[i] == torsion
    #assign params[i:i+4] to ff
    #use size dicts to correctly index into aligned ff
    i = 0
    for i, prm in enumerate(params_list):
        if prm[1] == 1: # torsion = 1
            #add dihedral parameter index to params list to avoid having to find it every time
            #either have it in the params file or find it with argwhere
            prm_idx = prm[1] # this idx may change
            geo_idx = prm[0]
            #TODO: add dynamic cluster sizes here
            cluster_num = geo_idx % 40
            aligned_ff[cluster_num]["t_k"][geo_idx][prm_idx] = params[i]
            aligned_ff[cluster_num]["t_period"][geo_idx][prm_idx] = params[i+1]
            aligned_ff[cluster_num]["t_phase"][geo_idx][prm_idx] = params[i+2]
            aligned_ff[cluster_num]["scee"][geo_idx][prm_idx] = params[i+3]
            aligned_ff[cluster_num]["scnb"][geo_idx][prm_idx] = params[i+4]
            i += 5
    return

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