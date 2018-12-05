import sys
import os
from sequencing import utilities

lonestar_template = '''\
#!/bin/bash

#$ -N {job_name}
#$ -pe 12way {num_cores}
#$ -q normal
#$ -o {job_name}.o$JOB_ID
#$ -l h_rt={time}
#$ -V
#$ -M jah@ices.utexas.edu
#$ -m be
{hold_line}
#$ -cwd
#$ -A SeqMap1
'''

stampede_template = '''\
#!/bin/bash

#SBATCH -J {job_name}
#SBATCH -n {num_cores}
#SBATCH -p normal
#SBATCH -o {job_name}.o%j
#SBATCH -e {job_name}.e%j
#SBATCH -t {time}
{hold_line}
#SBATCH -A SeqMap1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jah@ices.utexas.edu
'''

common_template = '''\
module load launcher
module load samtools
module load bwa

export EXECUTABLE=$TACC_LAUNCHER_DIR/init_launcher 
export CONTROL_FILE={commands_file_name}
export WORKDIR=.
 
# Variable description:
#
#  EXECUTABLE     = full path to the job launcher executable
#  CONTROL_FILE   = text input file which specifies
#                   executable for each process
#                   (should be located in WORKDIR)
#  WORKDIR        = location of working directory

if [ ! -e $WORKDIR ]; then
    echo " "
    echo "Error: unable to change to working directory."
	echo "       $WORKDIR"
	echo " "
	echo "Job not submitted."
	exit
fi

if [ ! -f $EXECUTABLE ]; then
	echo " "
	echo "Error: unable to find launcher executable $EXECUTABLE."
	echo " "
	echo "Job not submitted."
	exit
fi

if [ ! -f $WORKDIR/$CONTROL_FILE ]; then
	echo " "
	echo "Error: unable to find input control file $CONTROL_FILE."
	echo " "
	echo "Job not submitted."
	exit
fi

cd $WORKDIR/
echo " WORKING DIR:   $WORKDIR/"

$TACC_LAUNCHER_DIR/paramrun $EXECUTABLE $CONTROL_FILE

echo " "
echo " Parameteric Job Complete"

{optional_finish}
python $HOME/projects/mutations/code/Parallel/tacc_email_error.py {job_name} {job_dir} {job_id_var}
'''

def create(job_name,
           full_commands_file_name,
           time='1:00:00',
           optional_finish='# No optional finish',
           hold_jid=None,
          ):
    job_dir, commands_file_name = os.path.split(full_commands_file_name)

    hostname = os.environ['HOSTNAME']
    if 'ls4' in hostname:
        cores_per_node = 12
        launcher_file_name = '{0}/launcher_{1}.sge'.format(job_dir, job_name)
        specific_template = lonestar_template
        hold_template = '#$ -hold_jid {0}'
        job_id_var = '$JOB_ID'
    elif 'stampede' in hostname:
        cores_per_node = 16
        launcher_file_name = '{0}/launcher_{1}.slurm'.format(job_dir, job_name)
        specific_template = stampede_template
        hold_template = '#SBATCH -d afterok:{0}'
        job_id_var = '$SLURM_JOB_ID'
    
    num_cores = utilities.line_count(full_commands_file_name)
    while num_cores % cores_per_node != 0:
        num_cores += 1

    if hold_jid:
        hold_line = hold_template.format(hold_jid)
    else:
        hold_line = '# Not holding'
    
    specific_part = specific_template.format(job_name=job_name,
                                             num_cores=num_cores,
                                             time=time,
                                             hold_line=hold_line,
                                            )
    common_part = common_template.format(job_name=job_name,
                                         job_dir=job_dir,
                                         commands_file_name=commands_file_name,
                                         optional_finish=optional_finish,
                                         job_id_var=job_id_var,
                                        )
    
    with open(launcher_file_name, 'w') as launcher_file:
        launcher_file.write(specific_part + common_part)

    return launcher_file_name
