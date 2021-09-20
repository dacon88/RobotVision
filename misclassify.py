# ================================================================== #
#                        Adversarial Example                         #
# ================================================================== #

def predict(model, x, labels):
    with torch.no_grad():
        output = model(x)  # 1000 values (predictions on 1,000 classes)

    # softmax will rescale outputs so that the sum is 1 and we
    # can use them as probability scores
    scores = torch.softmax(output, dim=1)

    # take top k predictions - accuracy is usually measured with top-5
    _, preds = output.topk(k=5, dim=1)

    # use the output as index for the labels list
    for label in preds[0]:
        predicted_label = labels[label.item()]
        score = scores[0, label.item()].item()
        print("ID: {:3d} Label: {:25s} Score: {:.2f}"
              "".format(label.item(), predicted_label, Decimal(score)))


predict(model, input_batch, labels)
imshow(input_batch[0])


def batch_transform(batch, transform):
    return transform(batch.squeeze()).unsqueeze(0)


def l2(x):
    return x.view(x.shape[0], -1).norm(p=2, dim=1)


def perturb_iterative(x, y, model, nb_iter, eps, eta, loss_fn,
                      transform, inverse_transform,
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


loss = torch.nn.CrossEntropyLoss()
target_label = torch.LongTensor([402])
x_adv = perturb_iterative(x=input_batch, y=target_label, model=model,
                          nb_iter=500, eps=5, eta=0.03, loss_fn=loss,
                          transform=normalize,
                          inverse_transform=inv_normalize,
                          clip_min=0.0,
                          clip_max=1.0)

predict(model, x_adv, labels)
imshow(x_adv[0])