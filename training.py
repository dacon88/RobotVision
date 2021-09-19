"""
training handles the modified AlexNet NN training process
"""

from alexnet_finetune import AlexnetFinetune
import torch



def main():

    # TODO: parse argument to script?
    state_file_name = "state_dict_model.pt"

    # Initialize network
    alexnet_ft = AlexnetFinetune()

    alexnet_ft.train_model()
    alexnet_ft.save_nn_state(state_file_name)


if __name__ == "__main__":
    main()
