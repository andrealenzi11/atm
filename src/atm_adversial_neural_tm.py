from typing import List, Tuple

import gensim
import numpy as np
import torch
import torch.autograd as autograd
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import Dataset

if torch.cuda.device_count() == 0:
    device = 'cpu'
else:
    device = 'cuda'


class TopicModelDataset(Dataset):
    """
        Build Dataset for Adversarial Topic Modeling (ATM)
    """

    def __init__(self,
                 corpus: List[Tuple[int, float]],
                 len_dict: int):
        self.corpus = corpus
        self.len_dict = len_dict

    def __len__(self):
        return len(self.corpus)

    def __getitem__(self, idx):
        sample = np.zeros(self.len_dict)
        index, value = zip(*self.corpus[idx])
        sample[list(index)] = list(value)
        # tf-idf scores (sample vector) of the current document d
        # are divided by the sum of all the tf-idf scores of document d
        sample = sample / sample.sum()
        return sample


class PrintLayer(nn.Module):
    """
        Debug layer that print the shape of the passed tensor
    """

    def __init__(self):
        super(PrintLayer, self).__init__()

    def forward(self, x):
        # Do your print / debug stuff here
        print(f"\t\t shape: {x.shape}")
        return x


class Generator(nn.Module):
    """
        Generator Network
    """

    def __init__(self, opt):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(in_features=opt['num_topics'],
                      out_features=opt['enc_mid_layer']),
            nn.LeakyReLU(),  # nn.PReLU(num_parameters=1, init=0.25),
            nn.BatchNorm1d(num_features=opt['enc_mid_layer']),  # nn.LayerNorm(normalized_shape=opt['enc_mid_layer']),
            nn.Linear(in_features=opt['enc_mid_layer'],
                      out_features=opt['vocab_size']),
            nn.Softmax(dim=1),  # dim=-1
        )

    def forward(self, z):
        return self.model(z)


class Discriminator(nn.Module):
    """
        Discriminator network D

        A higher D output means that the discriminator is prone to consider
        the input data as a real document (higher D output, higher realness for D)
    """

    def __init__(self, opt):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(in_features=opt['vocab_size'],
                      out_features=opt['dec_mid_layer']),
            nn.LeakyReLU(),
            # nn.LayerNorm(normalized_shape=opt['dec_mid_layer']),
            nn.Linear(in_features=opt['dec_mid_layer'],
                      out_features=1)
        )

    def forward(self, z):
        return self.model(z)


class GenerativeAdversarialNetworkTopicModel:
    """
        Adversarial Topic Modeling (ATM)
    """

    def __init__(self, hyperparams_diz: dict):
        """
            Initialize generator and discriminator
        """
        self.hyperparams_diz = hyperparams_diz
        self.generator = Generator(self.hyperparams_diz).to(device)
        self.discriminator = Discriminator(self.hyperparams_diz).to(device)
        self.generator.apply(self.init_weights)
        self.discriminator.apply(self.init_weights)

    @staticmethod
    def init_weights(m):
        """
            Initialize network parameters
        """
        if type(m) == nn.Linear:
            nn.init.kaiming_normal_(m.weight)

    def compute_gradient_penalty(self, real_samples, fake_samples) -> torch.Tensor:
        """
            Calculates the gradient penalty loss for WGAN-GP

            from paper: "Improved Training of Wasserstein GANs".
        """
        # === Random weight term for interpolation between real and fake samples === #
        # epsilon = torch.FloatTensor(np.random.random(size=(real_samples.size(0), 1))).to(device)
        epsilon = torch.rand(size=(real_samples.size(0), 1)).to(device)
        epsilon = epsilon.expand(real_samples.size()).to(device)
        # === Get random interpolation between real and fake samples and pass it to discriminator === #
        interpolates = epsilon * real_samples + ((1 - epsilon) * fake_samples)
        interpolates = interpolates.requires_grad_(True).to(device)
        disc_interpolates = self.discriminator(interpolates)
        # === Tensor of 1 and shape (batch_size, 1) === #
        # ones_tensor = torch.FloatTensor(real_samples.shape[0], 1).fill_(value=1.0).requires_grad_(True).to(device)
        ones_tensor = torch.ones(disc_interpolates.size()).to(device)
        # === Get gradient tensor === #
        gradients = autograd.grad(
            outputs=disc_interpolates,
            inputs=interpolates,
            grad_outputs=ones_tensor,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        # === Compute and return Gradient Penalty === #
        gradient_penalty = ((gradients.norm(p=2, dim=1) - 1) ** 2).mean()
        return gradient_penalty

    def fit(self,
            dataloader,
            learning_rate_generator: float = 0.0001,
            betas_generator: tuple = (0, 0.9),
            learning_rate_discriminator: float = 0.0001,
            betas_discriminator: tuple = (0, 0.9)):
        """
            Fit on data, trying to find the optimal neural networks parameters
        """
        optimizer_generator = torch.optim.Adam(self.generator.parameters(),
                                               lr=learning_rate_generator,
                                               betas=betas_generator)

        optimizer_discriminator = torch.optim.Adam(self.discriminator.parameters(),
                                                   lr=learning_rate_discriminator,
                                                   betas=betas_discriminator)

        # ===== Iterate over Epochs ===== #
        for epoch in range(self.hyperparams_diz['n_epochs']):

            wasserstein_distances_epoch, w_cnt = torch.tensor(0.0).to(device), torch.tensor(0).to(device)
            discriminator_losses_epoch, d_cnt = torch.tensor(0.0).to(device), torch.tensor(0).to(device)
            generator_losses_epoch, g_cnt = torch.tensor(0.0).to(device), torch.tensor(0).to(device)
            num_batches = len(dataloader)

            # ===== Iterate over batches ===== #
            for batch_id, real_data in enumerate(dataloader):

                real_data = real_data.float().to(device)

                # reset the discriminator gradient
                optimizer_discriminator.zero_grad()

                # Sample noise tensor z from the dirichlet prior as generator input
                # - METHOD 1 (all rows tensors equals)
                dirichlet = torch.distributions.dirichlet.Dirichlet(
                    torch.FloatTensor([1 / self.hyperparams_diz['num_topics']
                                       for _ in range(self.hyperparams_diz['num_topics'])])
                )
                sample = dirichlet.sample()  # 1D tensor ('num_topics', 1)
                z = Variable(sample.repeat(real_data.shape[0], 1)).to(device)  # 2D tensor ('num_batches', 'num_topics')
                # - METHOD 2 (all rows tensors different)
                # z_numpy = np.random.dirichlet(
                #     [1 / hyperparams_diz['num_topics'] for _ in range(hyperparams_diz['num_topics'])],
                #     hyperparams_diz['batch_size']
                # ).astype(np.float32)
                # z = Variable(torch.from_numpy(z_numpy)).to(device)  # 2D tensor of shape ('num_batches', 'num_topics')

                # Generate a batch of fake data
                fake_data = self.generator(z)

                # Compute with the discriminator the real validity scores starting from the sampled real batch data
                real_validity_batch_scores = self.discriminator(real_data)

                # Compute with the discriminator the fake validity scores starting from the generated fake batch data
                fake_validity_batch_scores = self.discriminator(fake_data)

                # Compute Gradient Penalty
                gradient_penalty = self.compute_gradient_penalty(real_data.data, fake_data.data)

                # Compute the Discriminator_Loss: D(d_fake) - D(d_real) + (lambda * Gradient_Penalty)
                discriminator_loss = torch.mean(fake_validity_batch_scores) - \
                                     torch.mean(real_validity_batch_scores) + \
                                     self.hyperparams_diz['lambda_gp'] * gradient_penalty

                discriminator_losses_epoch += discriminator_loss
                d_cnt += 1

                # Optimize the discriminator loss
                discriminator_loss.backward()
                optimizer_discriminator.step()

                # reset the generator gradient
                optimizer_generator.zero_grad()

                # Train the Generator every 'n_critic' steps
                if batch_id % self.hyperparams_diz['n_critic'] == 0:
                    # Generate a batch of fake data
                    fake_data_2 = self.generator(z)
                    # Compute with the discriminator the fake validity scores from the generated fake data
                    fake_validity_batch_scores_2 = self.discriminator(fake_data_2)
                    # Compute the Generator Loss (measures generator's ability to fool the discriminator): -D(d_fake)
                    generator_loss = -torch.mean(fake_validity_batch_scores_2)
                    generator_losses_epoch += generator_loss
                    g_cnt += 1
                    # Optimize the generator loss
                    generator_loss.backward()
                    optimizer_generator.step()

                wasserstein_d = -torch.mean(real_validity_batch_scores) + torch.mean(fake_validity_batch_scores)
                wasserstein_distances_epoch += wasserstein_d
                w_cnt += 1
                # ===== end of iteration ever single batch ===== #

            print(
                f"[Epoch {epoch + 1}/{self.hyperparams_diz['n_epochs']}] "
                f"[#Batches {num_batches}] "
                f"[D loss: {discriminator_losses_epoch / d_cnt}] "
                f"[G loss: {generator_losses_epoch / g_cnt}] "
                f"[Wasserstein Distance: {wasserstein_distances_epoch / w_cnt}] "
            )
            # ===== end of iteration over single epoch ===== #

    def extract_topics(self,
                       dictionary: gensim.corpora.dictionary.Dictionary,
                       n_top_words: int = 50) -> List[List[str]]:
        """
            Inference of topics (clusters of words)

            EX. If vocabulary_size=1000 and num_topics=5:

                starting from the following input:
                [
                    [[1, 0, 0, 0, 0]],
                    [[0, 1, 0, 0, 0]],
                    [[0, 0, 1, 0, 0]],
                    [[0, 0, 0, 1, 0]],
                    [[0, 0, 0, 0, 1]],
                ]

                we generate in the variable 'emb' for each topic the words distribution:
                [
                    [w1_prob, w2_prob, ..., w_1000_prob],     # Topic 1
                    [w1_prob, w2_prob, ..., w_1000_prob],     # Topic 2
                    [w1_prob, w2_prob, ..., w_1000_prob],     # Topic 3
                    [w1_prob, w2_prob, ..., w_1000_prob],     # Topic 4
                    [w1_prob, w2_prob, ..., w_1000_prob],     # Topic 5
                ]

                we return the most probable words for each topic
        """
        self.generator.eval()

        with torch.no_grad():
            emb = [
                self.generator(
                    torch.FloatTensor([[
                        0 if i != j
                        else 1
                        for j in range(self.hyperparams_diz['num_topics'])
                    ]]).to(device)
                ).squeeze()
                for i in range(self.hyperparams_diz['num_topics'])]

            feature_names = list(zip(*sorted(dictionary.items(), key=lambda x: x[0])))[1]

            topics = [[feature_names[j] for j in emb[i].argsort(descending=True)[:n_top_words]]
                      for i in range(len(emb))]

            return topics
