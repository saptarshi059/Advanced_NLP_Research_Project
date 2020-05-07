printf "Checking if Anaconda is present on this system...\n"
if [ -x "$(command -v conda)" ]; then
	printf 'Anaconda is installed!' >&2
else
	printf "Anaconda is not present. Installing Anaconda...\n"
	wget https://repo.anaconda.com/archive/Anaconda3-2020.02-Linux-x86_64.sh
	bash Anaconda3-2020.02-Linux-x86_64.sh -b -p $HOME/anaconda3
	echo export PATH="$HOME/anaconda3/bin:$PATH" >> ~/.bashrc
	source ~/.bashrc
	rm Anaconda3-2020.02-Linux-x86_64.sh
fi

printf "Testing Installation...\n"
conda list	

printf "Creating Environment...\n"
conda env create -f environment.yml

printf "Testing Environment Setup...\n"
source activate resproj

printf "Deactivating Environment...\n"
conda deactivate