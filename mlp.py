import torch


class MLP(torch.nn.Module):

    def __init__(self, input_shape, output_shape, n_hidden_layer=1, hidden_layer_size=1024, *args, **kwargs):
        super().__init__(*args, **kwargs)
        model = [
            torch.nn.Linear(input_shape, hidden_layer_size),
            torch.nn.Tanh()
        ]

        for _ in range(n_hidden_layer):
            model.append(torch.nn.Linear(hidden_layer_size, hidden_layer_size))
            model.append(torch.nn.Tanh())

        model.append(torch.nn.Linear(hidden_layer_size, output_shape))
        model.append(torch.nn.Softmax(dim=1))

        self._model = torch.nn.Sequential(*model)

    def forward(self, x):
        return self._model.forward(x)
