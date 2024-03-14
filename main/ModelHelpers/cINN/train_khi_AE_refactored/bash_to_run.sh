cat << EOF > pyscript.py
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


input_command_def = "python3 /home/checkr99/InSituML/main/ModelHelpers/cINN/train_khi_AE_refactored/main.py "

input_command_def += " --property_ momentum_force "
input_command_def += " --learning_rate 1e-3 "
input_command_def += " --num_epochs 15 "
input_command_def += " --lossfunction earthmovers "
input_command_def += " --ae_config non_deterministic "
input_command_def += " --particles_to_sample 150000 "
input_command_def += " --project_kw momentum_force_emd "

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


encoder_kwargs = {'ae_config':'non_deterministic',
                  'fc_layer_config':'[256]',
                  'conv_add_bn':True}
                  
decoder_kwargs = {'initial_conv3d_size':'[16, 4, 4, 4]',
                  'add_batch_normalisation':True}

def extract_key(key, value):
    s="""" """
    
    if 'encoder_kwargs' in key:
        
        kys = key.split("::")[1]
        encoder_kwargs[kys] = value
        return 'encoder_kwargs', s + str(encoder_kwargs) + s
        
    if 'decoder_kwargs' in key:
        
        kys = key.split("::")[1]
        decoder_kwargs[kys] = value
        return 'decoder_kwargs',s + str(decoder_kwargs) + s
    
    return k, v

hp_dic_format = {'encoder_kwargs::conv_layer_config':['[16, 32, 64, 128, 256, 512]',
                                                      '[128, 256, 512, 1024]',
                                                      '[128, 256, 512, 1024, 2048]'],
                                      
                 'decoder_kwargs::initial_conv3d_size':['[16, 4, 4, 4]',
                                                        '[8, 8, 4, 4]',
                                                        '[8, 4, 8, 4]']
                 }

kwargs_out = generate_kwargs(hp_dic_format)

for kwgs_dic in kwargs_out:
    input_command = ""
    dir_name = ""
    for k, v in kwgs_dic.items():
        dir_name += f"{k}_{v}"
        k, v = extract_key(k,v)
        input_command += f" --{k} {v} "
        
    input_command = input_command_def + input_command
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
