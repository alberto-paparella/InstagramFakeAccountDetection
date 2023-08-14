# AI_PythonInMyBoot

This project is a deliverable created for the "Intelligenza Artificiale" exam from the Alma Mater Studiorum's Master's
Degree course in Computer Science.

## Installing and running the project

This section contains instructions to configure and run the project.

1. Either download or clone the project from GitLab.
2. Open up a terminal inside the project directory.
3. Create a virtual environment by running the command ``` python -m venv venv ```. This will create a virtual
   environment in which the libraries will get installed.
4. Install libraries with ``` ./venv/bin/pip install -r ./requirements.txt ```.
5. Run the script ``` main.py ``` with the command ``` ./venv/bin/python ./main.py ```.  

Once the script is running, follow the instructions on screen.

## Codebase structure

- `main.py` serves as executable script to run the experiments.
- `generate_dl_dataset.py` serves as executable script to instantiate or reset deep learning models.
- `utils/utils.py` contains many useful functions such as the ones to run the experiments or get the metric scores.
- `dataset/`
   - `normalizer.py` contains a script to create a single dataset from two different ones and export it in .json format.
   - `utils.py` contains many useful functions to work with the datasets, such as shuffling and splitting and getting combined datasets.
   - `deep/`
   - `sources/` contains the datasets that are being used for the experiments.
      - `automatedAccountData.json` contains the fake accounts of the `InstaFake` dataset.
      - `nonautomatedAccountData.json` contains the real accounts of the `InstaFake` dataset.
      - `user_fake_authentic_2class.csv` contains the `IJECE` dataset.
- `deep/`
   - `common.py`
   - `experiment.py`
   - `combined/`
   - `compatible/`
   - `IJECE/`
   - `InstaFake/`
- `visualization/`
   - `plotter.py` contains many useful functions to plot data and results.
   - `plots/` contains the plots for the data analysis on the original datasets.
   - `plots_results/` contains the plots representing the result of the experiments.
