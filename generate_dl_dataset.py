from dataset.utils import get_deep_learning_dataset

if __name__ == "__main__":
    a = input("Warning! This operation is destructive, and will invalidate all work on DL so far. \n"
              "Do you still want to proceed? (y/any) ")
    if a == "y":
        get_deep_learning_dataset()