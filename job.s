# -N job
#PBS -l nodes=1:ppn=9
#PBS -l walltime=3:00:00
#PBS -q normal
#PBS -l mem=200gb
#PBS -o /home/chenzy/eolog/result2
#PBS -e /home/chenzy/eolog/result2

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

for j in 512 
do
{
	for i in  1 0 2
	do
	{
		python -u test_reconstruction_method.py $i $j NGP &
		#python -u test_reconstruction_method.py $i $j CIC &
		#python -u test_reconstruction_method.py $i $j TSC &
	}
	done
	wait
}
done
wait
echo over








