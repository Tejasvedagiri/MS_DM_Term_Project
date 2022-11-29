import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, input_channels, hidden_dimesions, probalitiy_dimesions) -> None:
        super().__init__()

        self.linear = nn.Linear(input_channels, hidden_dimesions)
        self.linear2 = nn.Linear(hidden_dimesions, hidden_dimesions)
        self.hidden_mu = nn.Linear(hidden_dimesions, probalitiy_dimesions)
        self.hidden_sigma = nn.Linear(hidden_dimesions, probalitiy_dimesions)
    
        self.act = nn.ReLU()
    def forward(self, X):
        hidden = self.linear(X)
        hidden = self.linear2(hidden)
        hidden = self.act(hidden)

        mu = self.hidden_mu(hidden)
        sigma = self.hidden_sigma(hidden)

        return mu, sigma

class Decoders(nn.Module):
    def __init__(self, input_channels,  hidden_dimesions, probalitiy_dimesions) -> None:
        super().__init__()

        self.hidden = nn.Linear(probalitiy_dimesions, hidden_dimesions)
        self.linear2 = nn.Linear(hidden_dimesions, hidden_dimesions)
        self.out = nn.Linear(hidden_dimesions, input_channels)

        self.act = nn.ReLU()
    
    def forward(self, X):
        out = self.hidden(X)
        out = self.linear2(out)
        out = self.act(out)

        out = self.out(out)

        return out

class Model(nn.Module):
    def __init__(self, input_channels,  hidden_dimesions = 200, probalitiy_dimesions=20) -> None:
        super().__init__()

        self.encoder = Encoder(input_channels=input_channels, hidden_dimesions=hidden_dimesions, probalitiy_dimesions=probalitiy_dimesions)

        self.decoders = Decoders(input_channels=input_channels, hidden_dimesions=hidden_dimesions, probalitiy_dimesions=probalitiy_dimesions)

    def forward(self, X):
        mu, sigma = self.encoder(X)
        
        #Play around with eps
        eps = torch.randn_like(sigma, requires_grad=True)

        new_out = mu + (sigma * eps)

        new_X = self.decoders(new_out)

        return new_X, mu, sigma

def unit_test():
    X = torch.rand((5, 29))
    vae = Model(input_channels=X.shape[1], hidden_dimesions=256, probalitiy_dimesions=20)
    new_X, mu, sigma = vae(X)

    print(new_X.shape)
    print(mu.shape)
    print(sigma.shape)

    print(new_X)

if __name__=="__main__":
    unit_test()

