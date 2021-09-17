"""
training handles the modified AlexNet NN training process
"""

from alexnet_finetune import AlexnetFinetune


def main():
    alexnet_finetune = AlexnetFinetune()

    alexnet_finetune.train_model()


if __name__ == "__main__":
    main()
