# Getting everything working

1. Install Anaconda from website https://www.anaconda.com/products/individual

2. **Optional, but recommended!** Create a isolated environment for python. This allows you to keep the multitude of packages "tidy" :)

	***Documentation can be found @ https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html***

	2.1 Start Anaconda Promt (anaconda3) -> Search for it
	2.2 Type the following command: conda create --name myenv
		2.2.1 Replace myenv with a name of your choice, e.g. MyMasters
	2.3 When conda asks you to proceed, type y
	2.4 Now you simply need to activate your new environment, do this with: conda activate myenv

	*Note* You need to activate your environment every time you wish to use is, thus after installation useage will be Step 2.1 then 2.4

	2.5 Your new environment has no packages installed yet, the only manual installation needed for now is jupyter notebook. Install with:
		conda install jupyter notebook
	2.6 After succesfull installation, proceed to step 3.

3. Run jupyter notebook from the Anaconda Prompt with the following command: jupyter notebook

4. To install all other relevant packages, navigate to the LungSegmentation folder from the jupyter notebook interface.

5. Open the file Packages.ipynb using jupyter notebook, click on the first cell that reads print("Testing").
	5.1 Run the cell using Shift+Enter. You should see printed at the bottom of the cell Testing!

6. Run the second cell, which will install all relevant packages

7. Using jupyter open FanieScript_v1.ipynb

8. Use the script