cat << EOF > pyscript.py
from collections import OrderedDict
import numpy as np
import itertools
import os

slurm_content = "#!/bin/sh\n"
slurm_content += "#SBATCH -o %j.out\n"
slurm_content += "#SBATCH -e %j.err\n"
slurm_content += "#SBATCH --job-name={jobname}\n"
slurm_content += "#SBATCH --ntasks-per-node=1\n"
slurm_content += "#SBATCH --cpus-per-task=12\n"
slurm_content += "#SBATCH --account=casus\n"
slurm_content += "#SBATCH --partition={partition_name}\n"
slurm_content += "#SBATCH --gres=gpu:1\n"
slurm_content += "#SBATCH --time=48:00:00\n"
slurm_content += "module load python cuda gcc\n"
slurm_content += "source /home/checkr99/.new_env3.10/bin/activate\n\n"

def generate_kwargs(hyperparam_dic):

    numkeys = len(hyperparam_dic.keys())
    flatten_key_vals = ([[kys, vals] for kys, val_list in hyperparam_dic.items() for vals in val_list])
    
    kwargs_out = []
    
    for combs in itertools.combinations(np.arange(len(flatten_key_vals)), numkeys):

        kys = np.array(flatten_key_vals)[list(combs)][:, 0]

        if len(set(kys)) == len(kys):
            kwargs = {flatten_key_vals[i][0]: flatten_key_vals[i][1] for i in combs}
            kwargs_out.append(kwargs)
        else:
            continue
    
    return kwargs_out

hp_dic_format = {'learning_rate':[1e-3, 1e-4],
                 'z_dim':[1024, 2048],
                 'network':["VAE", "conAE"]}

kwargs_out = generate_kwargs(hp_dic_format)

for kwgs_dic in kwargs_out:
    input_command = "python3 main.py "
    dir_name = ""
    for k, v in kwgs_dic.items():
        input_command += f" --{k} {v} "
        dir_name += f"{k}_{v}"
    
    os.mkdir(dir_name)
    f = open(f"{dir_name}/job.sh", "w")
    f.write(slurm_content.format(jobname="Training", partition_name="casus"))
    f.write(input_command)

EOF

python3 pyscript.py

for dirs in */
do
 cd $dirs
 sbatch job.sh
 cd ..
done
