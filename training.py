"""
training handles the modified AlexNet NN training process
"""

from alexnet_finetune import AlexnetFinetune
import torch



def main():
    alexnet_finetune = AlexnetFinetune()

    alexnet_finetune.train_model()

    # Specify a path
    PATH = "state_dict_model.pt"
    #TODO: transform save into method
    # Save
    torch.save(alexnet_finetune.model.state_dict(), PATH)

if __name__ == "__main__":
    main()
