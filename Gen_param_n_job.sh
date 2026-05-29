#!/bin/bash

for cell in *.cell; do
    [ -e "$cell" ] || { echo "No .cell files found."; exit 1; }

    seed="${cell%.cell}"

    cat > "${seed}.param" << EOF
continuation : default
task : singlepoint
xc_functional : PBEsol
cut_off_energy : 1000 eV
spin_polarized : true
CALCULATE_POLARISATION : TRUE
#elec_energy_tol : 1e-5 eV

max_scf_cycles : 400
fix_occupancy : true
opt_strategy : speed

SPIN_TREATMENT : SCALAR
spin_orbit_coupling : false
EOF

    cat > "${seed}_job.sh" << EOF
#!/bin/bash
#SBATCH -J ${seed}
#SBATCH -n 24
#SBATCH -c 4
#SBATCH -p shared
#SBATCH -t 48:00:00

module purge
module load castep/26.1
mpirun castep.mpi ${seed}
EOF

    chmod +x "${seed}_job.sh"

    echo "Generated ${seed}.param and ${seed}_job.sh"
done
