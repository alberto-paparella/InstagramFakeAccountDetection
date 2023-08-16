# This script serves as the main runner for the project.
from dataset.normalizer import json_importer_full, csv_importer_full
from dataset.utils import find_demarcator, get_combined_datasets
from utils.utils import experiment
from visualization.plotter import print_all_plots, result_plot
from deep.experiment import run_experiment as dl_experiment


def main():
    exp_list = ["dt", "rf", "svm", "nbb", "nbg", "lr", "mp"]
    print("Benvenut* nel launcher esperimenti del progetto d'esame del team 'PythonInMyBoot."
          "\nIndicare quali esperimenti effettuare sui dataset tra i seguenti:"
          "\n dt  - Decision Tree"
          "\n rf  - Random Forest"
          "\n svm - Support Vector Machine"
          "\n nbb - Naive Bayes (Bernuolli dist.)"
          "\n nbg - Naive Bayes (Gaussian dist.)"
          "\n lr  - Logistic Regression"
          "\n mp  - Multilayer Perceptron"
          )
    experiments = input("Inserire i codici degli esperimenti (due caratteri l'uno) separati da uno spazio, "
                        "oppure * per eseguirli tutti: ")
    if experiments == "":
        print("Non è stato scelto nessun esperimento.")
        return
    if experiments != "*":
        try:
            vals = experiments.split(" ")
            for element in vals:
                if element not in exp_list:
                    print(f"L'esperimento specificato ({element}) non è tra gli elementi indicati {exp_list}")
                    return
            exp_list = vals
        except Exception:
            print("La stringa inserita non è valida.")
            return
    n_iter = int(input("Indicare il numero di iterazioni per ciascun esperimento: "))

    fake_if = json_importer_full("./dataset/sources/automatedAccountData.json", True)
    correct_if = json_importer_full("./dataset/sources/nonautomatedAccountData.json", False)

    default_dataset = csv_importer_full("./dataset/sources/user_fake_authentic_2class.csv")
    idx = find_demarcator(default_dataset)

    fake_IJECE = default_dataset[:idx]
    correct_IJECE = default_dataset[idx:]

    combined_dataset = get_combined_datasets()

    print('Salvataggio delle rappresentazioni delle caratteristiche dei dati...')
    print_all_plots(fake_if, correct_if, fake_IJECE, correct_IJECE)
    print('Grafici dei dati disponibili.')

    results = {
        "IJECEPaper":
            {
                'dt': {
                    'default': {'accuracy': 0.8834, 'precision': 0.886, 'recall': 0.883, 'f1': 0.883},
                    'custom': {'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
                },
                'rf': {
                    'default': {'accuracy': 0.9009, 'precision': 0.907, 'recall': 0.901, 'f1': 0.901},
                    'custom': {'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
                },
                'svm': { # Not used
                    'default': {'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0},
                    'custom': {'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
                },
                'nbb': { # Not used
                    'default': {'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0},
                    'custom': {'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
                },
                'nbg': {
                    'default': {'accuracy': 0.7312, 'precision': 0.759, 'recall': 0.731, 'f1': 0.724},
                    'custom': {'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
                },
                'lr': {
                    'default': {'accuracy': 0.8094, 'precision': 0.81, 'recall': 0.809, 'f1': 0.809},
                    'custom': {'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
                },
                'mp': {
                    'default': {'accuracy': 0.8173, 'precision': 0.818, 'recall': 0.817, 'f1': 0.817},
                    'custom': {'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
                }
            },
        "IFPaper":
            # In the InstaFake paper, they only take into consideration precision, recall and f1-score
            {
                'dt': {  # Not used
                    'default': {'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0},
                    'custom': {'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
                },
                'rf': {  # Not used
                    'default': {'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0},
                    'custom': {'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
                },
                'svm': {
                    'default': {'accuracy': 0.0, 'precision': 0.91, 'recall': 0.82, 'f1': 0.86},
                    'custom': {'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
                },
                'nbb': {
                    'default': {'accuracy': 0.0, 'precision': 0.85, 'recall': 0.68, 'f1': 0.78},
                    'custom': {'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
                },
                'nbg': {
                    'default': {'accuracy': 0.0, 'precision': 0.51, 'recall': 0.98, 'f1': 0.67},
                    'custom': {'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
                },
                'lr': {
                    'default': {'accuracy': 0.0, 'precision': 0.80, 'recall': 0.70, 'f1': 0.75},
                    'custom': {'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
                },
                'mp': {
                    'default': {'accuracy': 0.0, 'precision': 0.89, 'recall': 0.84, 'f1': 0.86},
                    'custom': {'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
                }
            },
        "InstaFake": dict(), "IJECE": dict(), "CompIJECE": dict(), "CompInstaFake": dict(),
        "ComboPar": dict(), "ComboFull": dict()
    }


    for exp in exp_list:
        print(
            "\nRunning test on dataset 'Instagram Fake and Automated Account Detection' (internal name: 'InstaFake')...")
        if exp == "mp":
            res = dl_experiment("./deep/InstaFake/checkpoint",
                                ["INSTAFAKE_DEFAULT_1692011874.977522", "INSTAFAKE_CUSTOM_1692011874.977522"],
                                "if", n_iter)
        else:
            res = experiment(fake_if, correct_if, csv=False, model=exp, n_iter=n_iter)
        results["InstaFake"][exp] = res
        print("\nRunning test on dataset 'IJECE' (internal name: 'IJECE')...")
        if exp == "mp":
            res = dl_experiment("./deep/IJECE/checkpoint",
                                ["IJECE_DEFAULT_1692120663.149388", "IJECE_CUSTOM_1692120054.02869"], "ijece", n_iter)
        else:
            res = experiment(fake_IJECE, correct_IJECE, csv=True, model=exp, n_iter=n_iter)
        results["IJECE"][exp] = res
        print("\nRunning test on dataset 'Compatibile - InstaFake' (internal name: 'CompInstaFake')...")
        if exp == "mp":
            res = dl_experiment("./deep/compatible/checkpoint",
                                ["", "COMP_INSTAFAKE_1692011881.013768"], "comp-if", n_iter)
        else:
            res = experiment(fake_if, correct_if, csv=False, model=exp, n_iter=n_iter, compatibility=True)
        results["CompInstaFake"][exp] = res

        print("\nRunning test on dataset 'Compatibile - IJECE' (internal name: 'CompIJECE')...")
        if exp == "mp":
            res = dl_experiment("./deep/compatible/checkpoint",
                                ["", "COMP_IJECE_1692011881.013768"], "comp-ijece", n_iter)
        else:
            res = experiment(fake_IJECE, correct_IJECE, csv=True, model=exp, n_iter=n_iter, compatibility=True)
        results["CompIJECE"][exp] = res
        print("\nRunning test on dataset 'Combo - Partial' (internal name: 'ComboPar')...")
        if exp == "mp":
            res = dl_experiment("./deep/combined/checkpoint",
                                ["", "COMBO_PART_1692011882.555224"], "combo-par", n_iter)
        else:
            res = experiment(combined_dataset["partial"]["fake"], combined_dataset["partial"]["correct"],
                             csv=False, model=exp, n_iter=n_iter, combine=True)
        results["ComboPar"][exp] = res
        print("\nRunning test on dataset 'Combo - Full' (internal name: 'ComboFull')...")
        if exp == "mp":
            res = dl_experiment("./deep/combined/checkpoint",
                                ["", "COMBO_FULL_1692011882.555224"], "combo-full", n_iter)
        else:
            res = experiment(combined_dataset["full"]["fake"], combined_dataset["full"]["correct"],
                             csv=False, model=exp, n_iter=n_iter, combine=True)
        results["ComboFull"][exp] = res

    # datasets = ["InstaFake", "IJECE", "CompInstaFake", "CompIJECE", "ComboPar", "ComboFull"]
    print(results)
    print('Salvataggio delle rappresentazioni dei risultati...')
    result_plot(results, exp_list, n_iter)
    print('Grafici dei risultati disponibili.')
    print("Esecuzione completata.")


if __name__ == "__main__":
    main()
