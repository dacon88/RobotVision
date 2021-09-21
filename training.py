"""
training handles the modified AlexNet NN training process
"""

from alexnet_finetune import AlexnetFinetune


def main():

    # TODO: parse argument to script?
    state_file_name = "state_dict_model.pt"

    # Top level data directory. Here we assume the format of the directory conforms
    #   to the ImageFolder structure
    dataset_dir = "/home/davide/university/iCubWorld"

    # img size, assume img has dimensions: img_size * img_size
    img_size = 160

    # Batch size for training (change depending on how much memory you have)
    batch_size = 8

    # Number of epochs to train for
    num_epochs = 3

    # Initialize network
    alexnet_ft = AlexnetFinetune(img_size, "train")

    alexnet_ft.train_model(dataset_dir, batch_size, num_epochs)
    alexnet_ft.save_nn_state(state_file_name)


if __name__ == "__main__":
    main()
