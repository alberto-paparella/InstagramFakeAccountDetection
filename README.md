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

- `main.py` serves as executable script to run the experiments
- `generate_dl_dataset.py` serves as executable script to instantiate or reset deep learning models
- `dataset/`
   - `normalizer.py`
   - ` utils.py`
   - `deep/`
   - `sources/`
   - `visualization/`
      - `plotter.py`
      - `plots/`
      - `plots_results`
   - `deep/`
      - `common.py`
      - `experiment.py`
      - `combined/`
      - `compatible/`
      - `IJECE/`
      - `InstaFake/`
   - `utils/utils.py` contains many useful functions such as the ones to run the experiments or get the metric scores.
