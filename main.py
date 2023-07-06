from dataset.normalizer import json_importer_full, csv_importer_full
from dataset.utils import find_demarcator, get_combined_datasets
from dataset.visualization.plotter import print_all_plots
from utils.utils import experiment


def main():
    exp_list = ["dt", "lr", "nb", "rf", "dl"]
    print("Benvenut* nel launcher esperimenti del progetto d'esame del team 'PythonInMyBoot."
          "\nIndicare quali esperimenti effettuare sui dataset tra i seguenti:"
          "\n dt - Decision Tree"
          "\n lr - Logistic Regression"
          "\n nb - Naive Bayes"
          "\n rf - Random Forest"
          "\n dl - Percettrone Multistrato"
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
    n_iter = int(input("Quante volte devono venire ripetuti gli esperimenti? "))
    fake_if = json_importer_full("./dataset/sources/automatedAccountData.json", True)
    correct_if = json_importer_full("./dataset/sources/nonautomatedAccountData.json", False)

    default_dataset = csv_importer_full("./dataset/sources/user_fake_authentic_2class.csv")
    idx = find_demarcator(default_dataset)

    fake_IJECE = default_dataset[:idx]
    correct_IJECE = default_dataset[idx:]

    combined_dataset = get_combined_datasets()

    print('Salvataggio delle rappresentazioni delle caratteristiche...')
    print_all_plots(fake_if, correct_if, fake_IJECE, correct_IJECE)
    print('Grafici disponibili.')
    results = {"InstaFake": dict(), "IJECE": dict(), "ComboPar": dict(), "ComboFull": dict()}
    for exp in exp_list:
        if exp == "dl":
            continue
        print(
            "\nRunning test on dataset 'Instagram Fake and Automated Account Detection' (internal name: 'InstaFake')...")
        res = experiment(fake_if, correct_if, csv=False, mode=exp, n_iter=n_iter)
        results["InstaFake"][exp] = res
        print("Running test on dataset 'IJECE' (internal name: 'IJECE')...")
        res = experiment(fake_IJECE, correct_IJECE, csv=True, mode=exp, n_iter=n_iter)
        results["IJECE"][exp] = res
        print("Running test on dataset 'Combo - Partial' (internal name: 'ComboPar')...")
        res = experiment(combined_dataset["partial"]["fake"], combined_dataset["partial"]["correct"],
                         csv=False, mode=exp, n_iter=n_iter, combine=True)
        results["ComboPar"][exp] = res
        print("Running test on dataset 'Combo - Full' (internal name: 'ComboFull')...")
        res = experiment(combined_dataset["full"]["fake"], combined_dataset["full"]["correct"],
                         csv=False, mode=exp, n_iter=n_iter, combine=True)
        results["ComboFull"][exp] = res
    if "dl" in exp_list:
        pass

    datasets = ["InstaFake", "IJECE"]
    wins = {"custom": 0, "default": 0}
    for exp in exp_list:
        for combined_dataset in datasets:
            if results[combined_dataset][exp]["default"]["precision"] < results[combined_dataset][exp]["custom"][
                "precision"]:
                print(f"Nell'esperimento {exp} su dataset {combined_dataset}, "
                      f"l'approccio custom ha ottenuto una precision più alta.")
                wins["custom"] += 1
            else:
                print(f"Nell'esperimento {exp} su dataset {combined_dataset}, "
                      f"l'approccio custom ha ottenuto una precision più bassa.")
                wins["default"] += 1
            print(
                f"(Differenza di "
                f"{results[combined_dataset][exp]['default']['precision'] - results[combined_dataset][exp]['custom']['precision']} pts)")
            if results[combined_dataset][exp]["default"]["accuracy"] < results[combined_dataset][exp]["custom"][
                "accuracy"]:
                print(f"Nell'esperimento {exp} su dataset {combined_dataset}, "
                      f"l'approccio custom ha ottenuto una accuracy più alta.")
                wins["custom"] += 1
            else:
                print(f"Nell'esperimento {exp} su dataset {combined_dataset}, "
                      f"l'approccio custom ha ottenuto una accuracy più bassa.")
                wins["default"] += 1
            print(
                f"(Differenza di "
                f"{results[combined_dataset][exp]['default']['accuracy'] - results[combined_dataset][exp]['custom']['accuracy']} pts)")
    print(f"L'approccio custom ha registrato {wins['custom']} vittorie, l'approccio default {wins['default']}")
    print(results)


if __name__ == "__main__":
    main()
