"""
training handles the modified AlexNet NN training process
"""

from alexnet_finetune import AlexnetFinetune


def main():

    # TODO: parse argument to script?
    state_file_name = "state_dict_model.pt"

    # Top level data directory. Here we assume the format of the directory conforms
    #   to the ImageFolder structure
    # TODO: if dataset not present, it give an error. fix it
    dataset_dir = "/Users/davide/Documents/laurea_magistrale/second_semester_first_year/machine_learning/ML_project/iCubWorld"

    # Batch size for training (change depending on how much memory you have)
    batch_size = 8

    # Number of epochs to train for
    num_epochs = 15

    # Flag for feature extracting. When False, we finetune the whole model,
    #   when True we only update the reshaped layer params
    feature_extract = False

    # Initialize network
    alexnet_ft = AlexnetFinetune()

    alexnet_ft.train_model(dataset_dir, batch_size, num_epochs, feature_extract)
    alexnet_ft.save_nn_state(state_file_name)


if __name__ == "__main__":
    main()
