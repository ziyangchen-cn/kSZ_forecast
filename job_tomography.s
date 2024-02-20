# -N job_tomo
#PBS -l nodes=1:ppn=9
#PBS -l walltime=24:00:00
#PBS -q normal
#PBS -l mem=100gb
#PBS -o /home/chenzy/eolog/result4
#PBS -e /home/chenzy/eolog/result4

# <<< conda initialize <<<
module load anaconda/anaconda-mamba
echo startt
source activate py4

# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/opt/anaconda-mamba/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/opt/anaconda-mamba/etc/profile.d/conda.sh" ]; then
        . "/opt/anaconda-mamba/etc/profile.d/conda.sh"
    else
        export PATH="/opt/anaconda-mamba/bin:$PATH"
    fi
fi
unset __conda_setup

module load anaconda/anaconda-mamba
source activate py4

echo startt

cd /home/chenzy/code/kSZ_forecast/

for hod_rand in 200 300 400
do
{
	for i in  1 0 2 #redshift_bin
	do
	{
		python -u generate_tomography.py 512 $i 0 0 0 $hod_rand &
	}
	done
}
done
wait
echo over








