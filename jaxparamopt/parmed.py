from parmed.amber import AmberParm
from parmed.tools import change, addLJType, changeRadii, tiMerge

from parmed.tools import setBond, deleteBond, addDihedral, addPDB

# Documented commands (type help <topic>):
# ========================================
# EOF                 changeRedoxState  minimize        scale           
# HMassRepartition    checkValidity     netCharge       scee            
# OpenMM              defineSolvent     outCIF          scnb            
# add12_6_4           deleteBond        outPDB          setAngle        
# addAtomicNumber     deleteDihedral    outparm         setBond         
# addDihedral         deletePDB         parm            setMolecules    
# addExclusions       energy            parmout         setOverwrite    
# addLJType           go                printAngles     shell           
# addPDB              gromber           printBonds      source          
# cd                  help              printDetails    strip           
# chamber             history           printDihedrals  summary         
# change              interpolate       printFlags      tiMerge         
# changeLJ14Pair      listParms         printInfo       writeCoordinates
# changeLJPair        lmod              printLJMatrix   writeFrcmod     
# changeLJSingleType  loadCoordinates   printLJTypes    writeOFF        
# changeProtState     loadRestrt        printPointers 
# changeRadii         ls                quit   

# interface to parmed
# TODO add author credit to all files
# and doc strings for each file/class/function

# need to write handler based on parameter index to modify
# all major parameter types

# also need to figure out if a better way exists to deal with custom parameters

# def custom_per_atom_param
# def custom_mapped_param - must supply the parameter mapping or trasformation

# TODO look into pre-computing actions and then passing the actions to this function to speed up callbacks
# section in the documentation about persistent data?
# how to track progress of actions?
def parmed_action_callback_generator(params, params_list, prmtop_name):
    parm = AmberParm(prmtop_name)

    for param in params:
        case 0:

        case 1:
            #addDihedral
            #deleteDihedral

        case 2:
            action = addLJType(parm, "@1", radius=1.5, epsilon=0.5)
            action.execute()

        # actions.change
        # Changes the property of given atoms to a new value.
        # <property> can be CHARGE, MASS, RADII, SCREEN, ATOM_NAME, ATOM_TYPE, ATOM_TYPE_INDEX, or ATOMIC_NUMBER
        # does not change shake assignment

        # changeLJPair and changeLJSingleType are both useful
        # changeLJ14Pair
        # bespoke vs all combinations essentially

        #deleteBond
        #addBond

        #setAngle
        #setBond




    return

# useful for comparison and testing
def energy_eval():
    # actions.energy
    # can specify cutoff, ewald, dispersion, omm, applayer, platform
    # precision and decompose
    # loadCoordinates will be useful too, also good for conversions
    # also loadRestrt
    return

def parmed_action_callback_executor(params, params_list, actions):
    for action in actions:
        # TODO how to modify action once it is created?
        action.execute()
    return

def geo_min():
    #actions.minimize can do bfgs minimization
    return

# TODO interoperability between omm/off/amber/charmm/gromacs
# back and forth directly or through parmed?
def openmm_executor():
    #parmed.tools.actions.OpenMM()
    return

def amber_to_parmed():
    return

def openmm_to_parmed():
    return

def amber_to_openmm():
    return

def gromber():
    # can load gromacs system as amber system
    return

# e.g. for polarizability type parameter
parm = AmberParm("original.prmtop")

n_atoms = parm.ptr('natom')
values = [0.1] * n_atoms

# Add the new section
parm.parm_data['%FLAG ELECTRONIC_POLARIZABILITY'] = values
parm.formats['%FLAG ELECTRONIC_POLARIZABILITY'] = '%FORMAT(5E16.8)'
parm.flag_list.append('%FLAG ELECTRONIC_POLARIZABILITY')

# Write out modified prmtop
# TODO do you use this or do you need to do actions.parmout?
# also need to look into writefrcmod, this may be very useful
parm.write_parm("custom.prmtop")

# All arguments separate
action = addLJType(parm, "@1", "radius", 1.5, "epsilon", 0.5)

# Also equivalent; keyword arguments given as keywords
action = addLJType(parm, "@1", radius=1.5, epsilon=0.5)

action.execute()