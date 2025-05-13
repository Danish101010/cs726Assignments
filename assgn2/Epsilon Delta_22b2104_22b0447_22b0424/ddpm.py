import torch
import torch.utils
import torch.utils.data
from tqdm.auto import tqdm
from torch import nn
import argparse
import torch.nn.functional as F
import utils
import dataset
import os
import matplotlib.pyplot as plt
import numpy as np



def embed(time, n_dim):
    
    time_embed = []
    # t= int(n_dim/2)
    for i in range(n_dim):
        if(i%2==0):
            time_embed.append(np.sin(1/10**(8*i/n_dim)*time))
        else:
            time_embed.append(np.cos(1/10**(8*i/n_dim)*time))

    return time_embed

class ErrorModel(nn.Module):
    def __init__(self, params):
        super(ErrorModel, self).__init__()
        self.model = nn.Sequential()

        k = params[0]
        l = len(params)

        for i, hidden_size in enumerate(params[1:l-1]):
            self.model.add_module(f"fc{i+1}", nn.Linear(k, hidden_size))
            self.model.add_module(f"relu{i+1}", nn.ReLU())
            k = hidden_size  

        self.model.add_module("fc_out", nn.Linear(k, params[-1]))

    def forward(self, x):
        x = self.model(x)
        return x
    
    
class NoiseScheduler():
    """
    Noise scheduler for the DDPM model

    Args:
        num_timesteps: int, the number of timesteps
        type: str, the type of scheduler to use
        **kwargs: additional arguments for the scheduler

    This object sets up all the constants like alpha, beta, sigma, etc. required for the DDPM model
    
    """
    def __init__(self, num_timesteps=50, type="linear", **kwargs):

        self.num_timesteps = num_timesteps
        self.type = type

        if type == "linear":
            self.init_linear_schedule(**kwargs)
        else:
            raise NotImplementedError(f"{type} scheduler is not implemented") # change this if you implement additional schedulers


    def init_linear_schedule(self, beta_start, beta_end):
        """
        Precompute whatever quantities are required for training and sampling
        """

        self.betas = torch.linspace(beta_start, beta_end, self.num_timesteps, dtype=torch.float32)

        self.alphas = 1 - self.betas
        self.alpha_bars = torch.ones_like(self.betas)

        self.alpha_bars[0] = self.alphas[0]
        for i in range(1, self.num_timesteps):
            self.alpha_bars[i] = self.alpha_bars[i - 1] * self.alphas[i]

    def init_cosine_schedule(self, beta_start, beta_end, s=0):
    
        steps = torch.linspace(0, 1, self.n_steps + 1)
        v_start = torch.cos(beta_start * torch.pi / 2) ** 2
        v_end = torch.cos(beta_end * torch.pi / 2) ** 2
        alphas = v_start + (v_end - v_start) * steps  
        betas = 1 - (alphas[1:] / alphas[:-1])  
        return torch.clamp(betas, min=0.0001, max=0.9999)  

    def init_sigmoid_schedule(self, beta_start, beta_end, s=6):

        steps = torch.linspace(-s, s, self.n_steps)  
        sigmoids = torch.sigmoid(steps)  
        betas = beta_start + (beta_end - beta_start) * sigmoids
        return torch.clamp(betas, min=0.0001, max=0.9999)  
   

    def __len__(self):
        return self.num_timesteps


class DDPM(nn.Module):
    def __init__(self, n_dim=3, n_steps=200):
        """
        Noise prediction network for the DDPM

        Args:
            n_dim: int, the dimensionality of the data
            n_steps: int, the number of steps in the diffusion process
        We have separate learnable modules for `time_embed` and `model`. `time_embed` can be learned or a fixed function as well

        """
        super(DDPM, self).__init__()

        self.time_embed = [embed(i, n_dim) for i in range(n_steps)]
        
        self.time_embed = torch.tensor(self.time_embed, dtype=torch.float32)

        self.model = ErrorModel([2*n_dim, 128,256,512,256, 128, n_dim])
        self.n_dim = n_dim

    def forward(self, x, t):
        """
        Args:
            x: torch.Tensor, the input data tensor [batch_size, n_dim]
            t: torch.Tensor, the timestep tensor [batch_size]

        Returns:
            torch.Tensor, the predicted noise tensor [batch_size, n_dim]
        """
        device = next(self.model.parameters()).device
        time_embed = self.time_embed.to(x.device)[t]
        # print(x.shape, time_embed.shape)
        if time_embed.dim() == 1:  
            time_embed = time_embed.unsqueeze(0).expand(x.shape[0], -1)

        x = torch.cat([x, time_embed], dim=-1)  
       
        return self.model(x.to(device))


    
class ConditionalDDPM(nn.Module):
    def __init__(self, n_classes = 8, n_dim=3, n_steps=200):
        """
        Class dependernt noise prediction network for the DDPM

        Args:
            n_classes: number of classes in the dataset
            n_dim: int, the dimensionality of the data
            n_steps: int, the number of steps in the diffusion process
        We have separate learnable modules for `time_embed` and `model`. `time_embed` can be learned or a fixed function as well

        """
        super().__init__()
        self.n_steps = n_steps
        self.n_classes = n_classes
        self.n_dim = n_dim
        embed_dim = 64
        
        # Time embed
        self.time_embed = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, embed_dim)
        )

        # Class embedding: n_classes + 1 for "unconditional" index
        self.class_embed = nn.Embedding(n_classes + 1, embed_dim)

        # Model that merges x + time_emb + class_emb

        self.model = ErrorModel([n_dim + embed_dim * 2,128,256,512,256, 128, n_dim])

    def forward(self, x, t, y):
        """
        Args:
            x: torch.Tensor, the input data tensor [batch_size, n_dim]
            t: torch.Tensor, the timestep tensor [batch_size]
            y: torch.Tensor, the class label tensor [batch_size]
        Returns:
            torch.Tensor, the predicted noise tensor [batch_size, n_dim]
        """
        if y is None:
            y = torch.full((x.size(0),), self.n_classes, device=x.device, dtype=torch.long)

        # time embedding
        t = t.float().unsqueeze(-1)
        t_emb = self.time_embed(t)

        # class embedding
        class_emb = self.class_embed(y)

        # concat
        inp = torch.cat([x, t_emb, class_emb], dim=1)
        return self.model(inp)  



# model, noise_scheduler, dataloader, optimizer, epochs, run_name

import torch.nn.functional as F

def trainCFG(model,noise_scheduler,dataloader,optimizer,epochs,run_name,cond_prob=0.9):
    
    device = next(model.parameters()).device
    model.train()
    model.to(device)

    alphas = noise_scheduler.alphas.to(device)        # shape [n_steps]
    betas = noise_scheduler.betas.to(device)          # shape [n_steps]
    alphas_cumprod = noise_scheduler.alpha_bars.to(device)  # shape [n_steps]
    n_steps = noise_scheduler.num_timesteps

    unconditional_index = model.n_classes  

    for epoch in range(epochs):
        epoch_loss = 0
        for x0, y in dataloader:
           
            x0 = x0.to(device)
            y  = y.to(device)

          
            t = torch.randint(0, n_steps, (x0.size(0),), device=device)

       
            noise = torch.randn_like(x0)  # same shape as x0
            alpha_t = alphas_cumprod[t].unsqueeze(-1)  # [batch_size, 1]
            alpha_t_sqrt = alpha_t.sqrt()
            one_minus_alpha_t_sqrt = (1 - alpha_t).sqrt()
            x_t = alpha_t_sqrt * x0 + one_minus_alpha_t_sqrt * noise
            mask = torch.rand(y.size(0), device=device) < cond_prob
            y_cond = y.clone()
            y_cond[~mask] = unconditional_index  
            y_cond = y_cond.long()
            pred_noise = model(x_t, t, y_cond)
            loss = F.mse_loss(pred_noise, noise)
            epoch_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1}/{epochs} | Avg.Loss: {epoch_loss/len(dataloader):.4f}")

        # Save model checkpoint each epoch
        torch.save(model.state_dict(), f"{run_name}/model.pth")



@torch.no_grad()
def sampleConditional(model, n_samples, noise_scheduler, guidance_scale=5.0, class_label=0):
 
    model.eval()
    device = next(model.parameters()).device


    alphas = noise_scheduler.alphas.to(device)
    betas = noise_scheduler.betas.to(device)    
    alphas_cumprod = noise_scheduler.alpha_bars.to(device)
    n_steps = noise_scheduler.num_timesteps
    
    alphas_cumprod_prev = torch.cat([torch.tensor([1.0], device=device), alphas_cumprod[:-1]], dim=0)
    n_dim = model.n_dim

    x = torch.randn(n_samples, n_dim, device=device)

    y = torch.full((n_samples,), class_label, device=device, dtype=torch.long)

    for i in reversed(range(n_steps)):
        t = torch.full((n_samples,), i, device=device, dtype=torch.long)

     
        eps_uncond = model(x, t, y=None) 
        eps_cond = model(x, t, y=y)

       
        eps = eps_uncond + guidance_scale * (eps_cond - eps_uncond)

        
        alpha_t = alphas_cumprod[i]
        alpha_t_prev = alphas_cumprod_prev[i]
        beta_t = betas[i]

        x = (1.0 / alpha_t.sqrt()) * (x - (1 - alpha_t) / (1 - alpha_t).sqrt() * eps)

        if i > 0:
       
            sigma_t = ((1 - alpha_t_prev)/(1 - alpha_t) * beta_t).sqrt()
            z = torch.randn_like(x)
            x = x + sigma_t * z

    return x


class ClassifierDDPM():
    """
    ClassifierDDPM implements a classification algorithm using the DDPM model
    """
    
    def __init__(self, model: ConditionalDDPM, noise_scheduler: NoiseScheduler):
        self.model = model
        self.noise_scheduler = noise_scheduler
        self.n_classes = model.n_classes



    def __call__(self, x):
        return self.predict(x)

    

    def predict_proba(self, x):
        """
        Args:
            x: torch.Tensor, the input data tensor [batch_size, n_dim]
        Returns:
            torch.Tensor, the predicted probabilites for each class  [batch_size, n_classes]
        """

        batch_size = x.shape[0]
        scores = torch.zeros(batch_size, self.n_classes, device=x.device)

        for c in range(self.n_classes):
            # class_label = torch.full((batch_size,), c, dtype=torch.long, device=x.device)
            recons = sampleConditional(self.model,n_samples= batch_size,noise_scheduler=self.noise_scheduler,class_label=c)  # Generate conditioned samples
            
            similarity = -F.mse_loss(x, recons, reduction='none').mean(dim=1) 
            scores[:, c] = similarity
        
        probs = F.softmax(scores, dim=1) 
        return probs
    def predict(self, x):
        """
        Args:
            x: torch.Tensor, the input data tensor [batch_size, n_dim]
        Returns:
            torch.Tensor, the predicted class tensor [batch_size]
        """

        probs = self.predict_proba(x)
        return torch.argmax(probs, dim=1)



def train(model, noise_scheduler, dataloader, optimizer, epochs, run_name):
    """
    Train the model and save the model and necessary plots

    Args:
        model: DDPM, model to train
        noise_scheduler: NoiseScheduler, scheduler for the noise
        dataloader: torch.utils.data.DataLoader, dataloader for the dataset
        optimizer: torch.optim.Optimizer, optimizer to use
        epochs: int, number of epochs to train the model
        run_name: str, path to save the model
    """
    criterion = nn.MSELoss()
    for epoch in range(epochs):
        model.train()  
        epoch_loss = 0
        
        for batch_X in dataloader:

            optimizer.zero_grad()

            # if isinstance(batch_X, tuple):  
            #     batch_X = batch_X[0]

            batch_X = batch_X[0].to(device)

            batch_size = batch_X.shape[0]
            t = torch.randint(0, noise_scheduler.num_timesteps, (batch_size,), device=device)

            eps = torch.randn_like(batch_X)
            alpha_bars_t = noise_scheduler.alpha_bars.to(device)[t].view(-1, 1)

            x_d = torch.sqrt(alpha_bars_t) * batch_X + torch.sqrt(1 - alpha_bars_t) * eps
            predictions = model(x_d, t)

            loss = criterion(predictions, eps)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {(epoch_loss/len(dataloader)):.4f}")
        torch.save(model.state_dict(), f'{run_name}/model.pth')


    



@torch.no_grad()
def sample(model, n_samples, noise_scheduler, return_intermediate=False): 
    """
    Sample from the model
    
    Args:
        model: DDPM
        n_samples: int
        noise_scheduler: NoiseScheduler
        return_intermediate: bool
    Returns:
        torch.Tensor, samples from the model [n_samples, n_dim]

    If `return_intermediate` is `False`,
            torch.Tensor, samples from the model [n_samples, n_dim]
    Else
        the function returns all the intermediate steps in the diffusion process as well 
        Return: [[n_samples, n_dim]] x n_steps
        Optionally implement return_intermediate=True, will aid in visualizing the intermediate steps
    """   

    
    device = next(model.parameters()).device 

    X_T = torch.randn(n_samples, model.n_dim, device=device)

    intermediate_steps = [] if return_intermediate else None

    for t in range(noise_scheduler.num_timesteps - 1, -1, -1):
        Z = torch.zeros_like(X_T)  
        if t > 0:
            Z = torch.randn_like(X_T) 

        alpha_bar_t = noise_scheduler.alpha_bars[t]
        alpha_t = noise_scheduler.alphas[t]
        
        eps_theta = model.forward(X_T, t)  

        
        X_T = (1 / torch.sqrt(alpha_t)) * (X_T - ((1 - alpha_t) / torch.sqrt(1 - alpha_bar_t)) * eps_theta) + torch.sqrt(1 - alpha_t) * Z  

        if return_intermediate:
            intermediate_steps.append(X_T.clone())

    return intermediate_steps if return_intermediate else X_T


def sampleCFG(model, n_samples, noise_scheduler, guidance_scale, class_label):
    """
    Sample from the conditional model
    
    Args:
        model: ConditionalDDPM
        n_samples: int
        noise_scheduler: NoiseScheduler
        guidance_scale: float
        class_label: int

    Returns:
        torch.Tensor, samples from the model [n_samples, n_dim]
    """
    pass

def sampleSVDD(model, n_samples, noise_scheduler, reward_scale, reward_fn):
    """
    Sample from the SVDD model

    Args:
        model: ConditionalDDPM
        n_samples: int
        noise_scheduler: NoiseScheduler
        reward_scale: float
        reward_fn: callable, takes in a batch of samples torch.Tensor:[n_samples, n_dim] and returns torch.Tensor[n_samples]

    Returns:
        torch.Tensor, samples from the model [n_samples, n_dim]
    """
    pass
    

if __name__ == "__main__":
    # print(torch.cuda.is_available())

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=['train', 'sample'], default='sample')
    parser.add_argument("--n_steps", type=int, default=None)
    parser.add_argument("--lbeta", type=float, default=None)
    parser.add_argument("--ubeta", type=float, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--n_samples", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--dataset", type=str, default = None)
    parser.add_argument("--seed", type=int, default = 42)
    parser.add_argument("--n_dim", type=int, default = None)

    args = parser.parse_args()
    utils.seed_everything(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    run_name = f'exps/ddpm_{args.n_dim}_{args.n_steps}_{args.lbeta}_{args.ubeta}_{args.dataset}' # can include more hyperparams
    os.makedirs(run_name, exist_ok=True)
    model = DDPM(n_dim=args.n_dim, n_steps=args.n_steps)
    noise_scheduler = NoiseScheduler(num_timesteps=args.n_steps, beta_start=args.lbeta, beta_end=args.ubeta)
    model = model.to(device)

    if args.mode == 'train':
        epochs = args.epochs
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        data_X,data_y = dataset.load_dataset(args.dataset)
        # can split the data into train and test -- for evaluation later
        data_X = data_X.to(device)
        dataloader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(data_X), batch_size=args.batch_size, shuffle=True)
        if data_y:
            data_y = data_y.to(device)
            dataloader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(data_X,data_y), batch_size=args.batch_size, shuffle=True)
            
        train(model, noise_scheduler, dataloader, optimizer, epochs, run_name)

    elif args.mode == 'sample':
        model.load_state_dict(torch.load(f'{run_name}/model.pth',weights_only=True))
        samples = sample(model, args.n_samples, noise_scheduler)
        torch.save(samples, f'{run_name}/samples_{args.seed}_{args.n_samples}.pth')
    else:
        raise ValueError(f"Invalid mode {args.mode}")