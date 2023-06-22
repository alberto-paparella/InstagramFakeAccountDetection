from dataset.normalizer import json_importer_full, csv_importer_full
from dataset.utils import find_demarcator
from utils.utils import experiment


def main():
    exp_list = ["dt", "lr", "nb", "rf", "dl"]
    print("Benvenut* nel launcher esperimenti del progetto d'esame del team 'PythonInMyBoot."
          "\nIndicare quali esperimenti effettuare sui dataset tra i seguenti:"
          "\n dt - Decision Tree"
          "\n lr - Linear Regression"
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
    fake_spz = json_importer_full("./dataset/sources/automatedAccountData.json", True)
    correct_spz = json_importer_full("./dataset/sources/nonautomatedAccountData.json", False)

    default_dataset = csv_importer_full("./dataset/sources/user_fake_authentic_2class.csv")
    idx = find_demarcator(default_dataset)

    fake_ijce = default_dataset[:idx]
    correct_ijce = default_dataset[idx:]
    results = {"spz": dict(), "IJCE": dict()}
    for exp in exp_list:
        print("\nRunning test on dataset 'Instagram Fake and Automated Account Detection' (internal name: 'spz')...")
        res = experiment(fake_spz, correct_spz, csv=False, mode=exp, n_iter=n_iter)
        results["spz"][exp] = res
        print("Running test on dataset 'IJCE' (internal name: 'IJCE')...")
        res = experiment(fake_ijce, correct_ijce, csv=True, mode=exp, n_iter=n_iter)
        results["IJCE"][exp] = res
    datasets = ["spz", "IJCE"]
    wins = {"custom": 0, "default": 0}
    for exp in exp_list:
        for dataset in datasets:
            if results[dataset][exp]["default"]["precision"] < results[dataset][exp]["custom"]["precision"]:
                print(f"Nell'esperimento {exp} su dataset {dataset}, "
                      f"l'approccio custom ha ottenuto una precision più alta.")
                wins["custom"] += 1
            else:
                print(f"Nell'esperimento {exp} su dataset {dataset}, "
                      f"l'approccio custom ha ottenuto una precision più bassa.")
                wins["default"] += 1
            print(
                f"(Differenza di "
                f"{results[dataset][exp]['default']['precision'] - results[dataset][exp]['custom']['precision']} pts)")
            if results[dataset][exp]["default"]["accuracy"] < results[dataset][exp]["custom"]["accuracy"]:
                print(f"Nell'esperimento {exp} su dataset {dataset}, "
                      f"l'approccio custom ha ottenuto una accuracy più alta.")
                wins["custom"] += 1
            else:
                print(f"Nell'esperimento {exp} su dataset {dataset}, "
                      f"l'approccio custom ha ottenuto una accuracy più bassa.")
                wins["default"] += 1
            print(
                f"(Differenza di "
                f"{results[dataset][exp]['default']['accuracy'] - results[dataset][exp]['custom']['accuracy']} pts)")
    print(f"L'approccio custom ha registrato {wins['custom']} vittorie, l'approccio default {wins['default']}")
    print(results)


if __name__ == "__main__":
    main()
