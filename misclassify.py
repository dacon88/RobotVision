from PIL import Image
import torch
from torchvision import transforms


from alexnet_finetune import AlexnetFinetune

transform = transforms.Compose([
    transforms.Resize(160),
    transforms.CenterCrop(160),
    transforms.ToTensor()
])

normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

# we need this later for bringing the image back to input space
inv_normalize = transforms.Normalize(
    mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
    std=[1 / 0.229, 1 / 0.224, 1 / 0.225])

def batch_transform(batch, transform):
    return transform(batch.squeeze()).unsqueeze(0)


def l2(x):
    return x.view(x.shape[0], -1).norm(p=2, dim=1)


def perturb_iterative(x, y, model, nb_iter, eps, eta, loss_fn, transform, inverse_transform,
                      clip_min=0.0, clip_max=1.0):
    """
    Iteratively maximize the loss over the input.

    :param x: input data (in feature space).
    :param y: target label.
    :param model: model to run forward pass function.
    :param nb_iter: number of iterations.
    :param eps: attack maximum l2 norm.
    :param eta: attack step size.
    :param loss_fn: loss function.
    :param transform: transformation to apply to the samples.
    :param inverse_transform: inverse process for bringing the samples
        back to input space.
    :param clip_min: mininum value per input dimension.
    :param clip_max: maximum value per input dimension.
    :return: tensor containing the perturbed input.
    """

    x_adv = x.clone()
    if torch.cuda.is_available():
        x_adv = x_adv.to('cuda')
        y = y.to('cuda')
        model = model.to('cuda')

    # track gradients on delta
    x_adv.requires_grad = True

    x_input_space = batch_transform(x, inverse_transform)  # x0

    for ii in range(nb_iter):
        outputs = model(x_adv)
        loss = loss_fn(outputs, y)
        loss.backward()

        grad = x_adv.grad.data
        x_adv.data = x_adv.data - eta * grad

        # ------------------------- PROJECTIONS --------------------------
        # now we have to enforce constraints in the input space
        x_adv_input_space = batch_transform(x_adv, inverse_transform)
        # box projection in input space
        x_adv_input_space = torch.clamp(x_adv_input_space, clip_min, clip_max)
        # l2 projection in input space
        if l2(x_adv_input_space - x_input_space) > eps:
            delta = x_adv_input_space.data - x_input_space.data
            delta = delta / l2(delta)
            x_adv_input_space.data = x_input_space.data + delta.data
        # project x_adv back onto feature/normalized space
        x_adv.data = batch_transform(x_adv_input_space, transform)
        # ----------------------------------------------------------------

        # reset gradient
        x_adv.grad.data.zero_()

    # we need to detach the gradient
    return x_adv.detach()


def main():
    nn = AlexnetFinetune()
    nn.model.load_state_dict(torch.load("state_dict_model.pt", map_location=torch.device('cpu')))
    nn.model.eval()
    nn.get_classes_names_from_csv("classes_names.csv")

    # not perturbed img classification
    single_img = "test_img/pepper.ppm"
    im = Image.open(single_img)
    im.show()
    prediction = nn.predict_image(im)
    print(prediction)

    # apply transform from torchvision
    input_tensor = normalize(transform(im))
    # create a mini-batch as expected by the model
    input_batch = input_tensor.unsqueeze(0)

    loss = torch.nn.CrossEntropyLoss()

    target_label = torch.LongTensor([7])  # index of correct prediction
    x_adv = perturb_iterative(x=input_batch, y=target_label, model=nn.model,
                              nb_iter=50, eps=5, eta=0.03, loss_fn=loss,  #nb_iter = 500
                              transform=normalize,
                              inverse_transform=inv_normalize,
                              clip_min=0.0,
                              clip_max=1.0)

    adv_img = transforms.ToPILImage()(x_adv[0])
    adv_img.show()

    prediction = nn.predict_image(adv_img)
    print(prediction)


if __name__ == "__main__":
    main()
